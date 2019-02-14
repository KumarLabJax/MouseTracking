import cv2
from .readers import means, scale
import numpy as np

# Plots the ellipse on the plot
def plot_ellipse(plot, label, color = (255, 255, 0)):
	label = np.add(np.multiply(label, scale), means)
	labelAngle = -np.arctan2(label[4],label[5])*180/np.pi
	if label[2] > 0 and label[3] > 0:
		cv2.ellipse(plot,(np.float32(label[0]),np.float32(label[1])),(np.float32(label[2]/2.0),np.float32(label[3]/2.0)),labelAngle,0.0,360.0,color)
		# Direction
		# (x,y) to (x+cos(angle)*maj/2,y+sin(angle)*maj/2)
		cv2.line(plot,(np.float32(label[0]),np.float32(label[1])),(np.float32(label[0]+label[4]*label[3]/2.0),np.float32(label[1]+label[5]*label[3]/2.0)),color)
	return plot

# Plots an xy hash on the plot
def plot_xy(plot, label, color = (255, 255, 0), rescale = True):
	if rescale:
		label = np.add(np.multiply(label, scale), means)
	cv2.line(plot,(np.float32(label[0]-2), np.float32(label[1])),(np.float32(label[0]+2), np.float32(label[1])), color)
	cv2.line(plot,(np.float32(label[0]), np.float32(label[1]-2)),(np.float32(label[0]), np.float32(label[1]+2)), color)
	return plot

# Merges the mask of a segmented image
def plot_seg(plot, label, color = 1):
	plot[:,:,color] = -cv2.resize(label,(480,480))+1
	return plot

def plot_image_seg(image, label):
	plot = cv2.cvtColor(image/255.0,cv2.COLOR_GRAY2RGB)
	plot = plot_seg(plot, label/255.0)
	cv2.imshow('PlotSeg',plot)
	cv2.waitKey()
	cv2.destroyAllWindows()

def plot_image_seg_compare(image, label, label2):
	plot = cv2.cvtColor(image/255.0,cv2.COLOR_GRAY2RGB)
	plot = plot_seg(plot, label/255.0)
	plot = plot_seg(plot, label2/255.0, 2)
	cv2.imshow('PlotSeg',plot)
	cv2.waitKey()
	cv2.destroyAllWindows()

# Plots the image with the given label
def plot_image_labels(image, label):
	plot = cv2.cvtColor(image/255.0,cv2.COLOR_GRAY2RGB)
	plot = plot_ellipse(plot, label)
	cv2.imshow('PlotEllipse',plot)
	cv2.waitKey()
	cv2.destroyAllWindows()

# Plots the image with the given label and label2
def plot_image_labels_compare(image, label, label2):
	plot = cv2.cvtColor(image/255.0,cv2.COLOR_GRAY2RGB)
	plot = plot_ellipse(plot, label) # Cyan default
	plot = plot_ellipse(plot, label2, (255, 0, 255)) # Magenta
	cv2.imshow('PlotEllipse',plot)
	cv2.waitKey()
	cv2.destroyAllWindows()

def plot_image_xy(image, label):
	plot = cv2.cvtColor(image/255.0,cv2.COLOR_GRAY2RGB)
	plot = plot_xy(plot, label) # Cyan default
	cv2.imshow('PlotXY',plot)
	cv2.waitKey()
	cv2.destroyAllWindows()

def plot_image_xy_compare(image, label, label2):
	plot = cv2.resize(cv2.cvtColor(image/255.0,cv2.COLOR_GRAY2RGB),(480,480))
	plot = plot_xy(plot, label) # Cyan default
	plot = plot_xy(plot, label2, (255, 0, 255)) # Magenta
	cv2.imshow('PlotXY',plot)
	cv2.waitKey()
	cv2.destroyAllWindows()

def save_image_labels(image, label, filename):
	plot = cv2.resize(cv2.cvtColor(image,cv2.COLOR_GRAY2RGB),(480,480))
	plot = plot_ellipse(plot, label) # Cyan default
	cv2.imwrite(filename, plot)

def save_image_labels_compare(image, label, label2, filename):
	plot = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
	plot = plot_ellipse(plot, label) # Cyan default
	plot = plot_ellipse(plot, label2, (255, 0, 255)) # Magenta
	cv2.imwrite(filename, plot)

def save_image_xy(image, label, filename):
	plot = cv2.resize(cv2.cvtColor(image,cv2.COLOR_GRAY2RGB),(480,480))
	plot = plot_xy(plot, label) # Cyan default
	cv2.imwrite(filename, plot)

def save_image_xy_compare(image, label, label2, filename):
	plot = cv2.resize(cv2.cvtColor(image,cv2.COLOR_GRAY2RGB),(480,480))
	plot = plot_xy(plot, label) # Cyan default
	plot = plot_xy(plot, label2, (255, 0, 255)) # Magenta
	cv2.imwrite(filename, plot)

def save_image_seg(image, label):
	plot = cv2.cvtColor(image/255.0 ,cv2.COLOR_GRAY2RGB, filename)
	plot = plot_seg(plot, label/255.0)
	cv2.imwrite(filename, plot*255.0)

def save_image_seg_compare(image, label, label2, filename):
	plot = cv2.cvtColor(image/255.0,cv2.COLOR_GRAY2RGB)
	plot = plot_seg(plot, label/255.0)
	plot = plot_seg(plot, label2/255.0, 2)
	cv2.imwrite(filename, plot*255.0)

