
# coding: utf-8

# In[1]:

input('Press <ENTER> to continue') 
# Standard scientific Python imports
# to install 
# conda install matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pylab import imshow, show, cm, savefig
input('Press <ENTER> to continue') 

# In[2]:


# Import datasets, classifiers and performance metrics
# to install
# conda install -c anaconda scikit-learn
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_mldata


# In[4]:


# The digits dataset
digits = fetch_mldata('MNIST original', data_home="MNIST_Cache")
mnist_image = np.asarray(digits.data)
mnist_label = np.asarray(digits.target)
mnist_image.shape


# In[5]:


# digits.data[1,:]
def view_image(image, label=""):
	"""View a single image."""
	print("Label: %s" % label)
	imshow(image, cmap=cm.gray)
	savefig(str(label)+".png")#saing images based upon their lable as their filename
	show()


# In[6]:


# images_and_labels = list(zip(digits.data, digits.target))
jay =mnist_image.reshape([70000,28,28])
# for index, (image, label) in enumerate(images_and_labels[:4]):
#     plt.subplot(2, 4, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Training: %i' % label)
# images_and_labels
view_image(jay[0,:,:], mnist_label[0])
input('Press <ENTER>') 

# In[7]:


# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.data)
# data = digits.data.reshape((n_samples, -1))
data = digits.data
data.shape
input('Press <ENTER> to continue') 

# In[8]:

print("Generating Classifier model")
# Create a classifier: a support vector classifier
classifier = svm.SVC(kernel="linear")
input('waiting for Classifier, paused') 

# In[ ]:


# We learn the digits on the first half of the digits
print("Training the model")
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
input('Press <ENTER> to continue') 

# In[ ]:


# Now predict the value of the digit on the second half:
print("Predicting from the model")
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])
input('Press <ENTER> to continue') 

# In[ ]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
input('Press <ENTER> to continue') 

# In[ ]:


print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
input('Press <ENTER> to continue') 

# In[ ]:


results = np.asarray(metrics.confusion_matrix(expected, predicted))
np.trace(results)/10#resultant accuracy
input('Press <ENTER> to continue') 
# images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))


# In[32]:


# for index, (image, prediction) in enumerate(images_and_predictions[:4]):
#     plt.subplot(2, 4, index + 5)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Prediction: %i' % prediction)
# plt.show()

