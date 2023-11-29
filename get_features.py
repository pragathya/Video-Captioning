import shutil
import tqdm
import numpy as np
import cv2
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model



class config:
    train_path = "data/training_data"
    test_path = "data/testing_data"
    batch_size = 126
    learning_rate = 0.0007
    epochs = 150
    latent_dim = 512
    num_encoder_tokens = 4096
    num_decoder_tokens = 1500
    time_steps_encoder = 80
    max_probability = -1
    save_model_path = 'models'
    validation_split = 0.15
    max_length = 10
    search_type = 'greedy'

def video_to_frames(video):
    path = os.path.join(config.test_path, 'temporary_images')
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    video_path = os.path.join(config.test_path, 'video', video)
    count = 0
    image_list = []
    # Path to video file
    cap = cv2.VideoCapture(video_path)
    frame_skip = 3  # Number of frames to skip

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        # if count % (frame_skip + 1) == 0:
        cv2.imwrite(os.path.join(config.test_path, 'temporary_images', 'frame%d.jpg' % count), frame)
        image_list.append(os.path.join(config.test_path, 'temporary_images', 'frame%d.jpg' % count))

        count += 1

    cap.release()
    cv2.destroyAllWindows()
    return image_list


def model_cnn_load():
    model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    out = model.layers[-2].output
    model_final = Model(inputs=model.input, outputs=out)
    return model_final


def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    return img


def extract_features(video, model):
    """

    :param video: The video whose frames are to be extracted to convert into a numpy array
    :param model: the pretrained vgg16 model
    :return: numpy array of size 4096x80
    """
    video_id = video.split(".")[0]
    print(video_id)
    print(f'Processing video {video}')

    image_list = video_to_frames(video)
    samples = np.round(np.linspace(
        0, len(image_list) - 1, 80))
    image_list = [image_list[int(sample)] for sample in samples]
    images = np.zeros((len(image_list), 224, 224, 3))

    for i in range(len(image_list)):
        img = load_image(image_list[i])
        images[i] = img


    images = np.array(images)

    try:
        fc_feats = model.predict(images, batch_size=32)
    except Exception as e:
        import traceback
        traceback.print_exc()

    img_feats = np.array(fc_feats)

    # cleanup
    shutil.rmtree(os.path.join(config.test_path, 'temporary_images'))

    return img_feats


def extract_feats_pretrained_cnn():
    """
    saves the numpy features from all the videos
    """
    model = model_cnn_load()
    print('Model loaded')

    if not os.path.isdir(os.path.join(config.test_path, 'feat')):
        os.mkdir(os.path.join(config.test_path, 'feat'))

    video_list = os.listdir(os.path.join(config.test_path, 'video'))
    
    #ًWhen running the script on Colab an item called '.ipynb_checkpoints' 
    #is added to the beginning of the list causing errors later on, so the next line removes it.

    try:
        video_list.remove('.ipynb_checkpoints')
    except: pass
    
    for video in video_list:

        outfile = os.path.join(config.test_path, 'feat', video + '.npy')
        img_feats = extract_features(video, model)
        np.save(outfile, img_feats)


if __name__ == "__main__":
    extract_feats_pretrained_cnn()
