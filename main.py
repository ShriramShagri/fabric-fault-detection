from src import *
from src.data_preprocessing import *
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import load_model

import sys, os

def trainModel(datasetPath):
    """Load dataset and train model

    Args:
        datasetPath (tuple): path of datasets

    Raises:
        InvalidDatasetError: When less than two directoris for dataset mentioned
    """    
    if len(datasetPath) < 2:
        raise InvalidDatasetError("0 or 1 dataset was passed please mention atleast two")
    
    if not (os.path.exists(datasetPath[0]) or os.path.exists(datasetPath[1])):
        raise InvalidDatasetError("Dataset path invalid")

    generateModel(datasetPath[0], datasetPath[1], overflow = (len(os.listdir(datasetPath[0])) > 2000 or len(os.listdir(datasetPath[1])) > 2000))

def generateTrainingSet(path, batchOfImages):
    """Manage batch of images

    Args:
        path (str): Directory of the test images
        batchOfImages (bool): batch of images or single image

    Returns:
        np array: array of images
    """    
    if batchOfImages:
        x = []
        for images in os.listdir(path):
            abspath = os.path.join(path, images)
            x.append(Preprocessor(abspath).filter())
    else:
        x = [Preprocessor(path).filter()]
    x = np.array(x)
    return x
    

def predict(path, weights = None, batchOfImages = False):
    try:
        if weights:
            model1 = load_model(weights[0])
            model1.load_weights(weights[1])
            model2 = load_model(weights[2])
            model2.load_weights(weights[3])

        else:
            model1 = load_model('./src/saved_models/detector_keras.h5')
            model1.load_weights('./src/saved_models/detector_weights.h5')
            model2 = load_model('./src/saved_models/classifier_keras.h5')
            model2.load_weights('./src/saved_models/classifier_weights.h5')

        model1.summary()

        x = generateTrainingSet(path, batchOfImages)

        test_datagen = ImageDataGenerator(rescale=1./255)

        i = 0
        text_labels = []
        plt.figure(figsize = (30, 20))
        columns = 5

        for batch in test_datagen.flow(x, batch_size=1):
            pred = model1.predict(batch)
            print(pred)
            if pred > 0.5:
                pred2 = model2.predict(batch)
                print(pred2)
                if pred2 > 0.5:
                    text_labels.append('Missing Thread')
                else:
                    text_labels.append('Hole/Stain')
            else:
                text_labels.append('No defect')
                
            plt.subplot(5/columns+1, columns, i+1)
            plt.title(text_labels[i])
            imgplot = plt.imshow(batch[0])
            i+=1
            if i%10 == 0:
                break
        
        if not batchOfImages:
            plt.subplot(5/columns+1, columns, 3)
            plt.title("Original")
            imgplot = plt.imshow(cv.cvtColor(cv.imread(path) , cv.COLOR_BGR2RGB))
        plt.show()
    except:
        raise PredictionError("Error during prediction")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1], batchOfImages=sys.argv[2] == 'True')
    else:
        predict('C:\\Users\\Shagri\\Desktop\\DIP\\Dataset\\Dataset_45\\Thread_missing\\6.jpg')