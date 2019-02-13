from ImageClassifier import ImageClassifier

#creating new model
IC = ImageClassifier('cat-dog-training_data', classifier_name='cat_dogs', batch_size=100)
# training model
IC.train(2000)
#testing this model
IC.test('cat-dog-testing_data')
# classify an image
prediction = IC.predict('predict-dog.jpg')
print(prediction)
