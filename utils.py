import os
import time
from statistics import mean


from matplotlib.image import imread
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim
import numpy as np
from skimage import io, util
from sklearn.feature_extraction import image
from ksvd import ApproximateKSVD
from PIL import Image
import cv2

ROOT_DIR = os.path.abspath(".")
ORG_IMAGE_DIR = os.path.join(ROOT_DIR, "original_images")

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = img_as_float(imread(os.path.join(folder,filename)))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def plot_images(all_or_one, filenames, images, random_image, random_filename, which_image):
    if all_or_one == 'all':
        for file, img in zip(filenames, images):
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'{which_image} image {file}')
            plt.show()
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(random_image)
        plt.axis('off')
        plt.title(f'Random {which_image} image {random_filename}')
        plt.show()

def add_gaussian_noise(image, sigma):
    noisy = random_noise(image, var=sigma**2)
    return noisy

def save_noisy_images(filenames, images, output_dir):
    sigmas = [0.1, 0.25, 0.5, 1, 0.08, 0.06, 0.04]
    for i in range(len(sigmas)):
        for file, img in zip(filenames, images):
            noisy = add_gaussian_noise(img, sigmas[i])
            split_file = file.split(".")[0]
            if i==0:
                noisy_path = f'{output_dir}/sig01/noisy_{split_file}_sig01.JPG'
                plt.imsave(noisy_path, noisy)
            elif i==1:
                noisy_path = f'{output_dir}/sig025/noisy_{split_file}_sig025.JPG'
                plt.imsave(noisy_path, noisy)
            elif i==2:
                noisy_path = f'{output_dir}/sig05/noisy_{split_file}_sig05.JPG'
                plt.imsave(noisy_path, noisy)
            elif i==3:
                noisy_path = f'{output_dir}/sig1/noisy_{split_file}_sig1.JPG'
                plt.imsave(noisy_path, noisy)
            elif i==4:
                noisy_path = f'{output_dir}/sig008/noisy_{split_file}_sig008.JPG'
                plt.imsave(noisy_path, noisy)
            elif i==5:
                noisy_path = f'{output_dir}/sig006/noisy_{split_file}_sig006.JPG'
                plt.imsave(noisy_path, noisy)
            elif i==6:
                noisy_path = f'{output_dir}/sig004/noisy_{split_file}_sig004.JPG'
                plt.imsave(noisy_path, noisy)

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def showStatistics(metric, value_of_sigma, results):
    avg = mean(results)
    maximum = max(results)
    minimum = min(results)
    print("Metryka", metric, "Dla wartości sigmy =",value_of_sigma, "\nMa średnią wynoszącą ", avg,
         "\nMaximum wynosi ", maximum, "\nMinimum ",minimum,"\n")

def showPlots(metric, results_004, results_01, results_006, results_008, results_025, results_05, results_1):
    plt.figure(figsize=(14,8))
    plt.plot(results_004, label="sigma 004", linestyle='--')
    plt.plot(results_006, label="sigma 006", linestyle='--')
    plt.plot(results_008, label="sigma 008", linestyle="--")
    plt.plot(results_01, label="sigma 01", linestyle="--")
    plt.plot(results_025, label="sigma 025", linestyle="--")
    plt.plot(results_05, label="sigma 05", linestyle="--")
    plt.plot(results_1, label="sigma 1", linestyle="--")

    plt.title(metric)
    plt.xlabel('Photo ID')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    plt.figure(figsize=(14,8))
    plt.plot(results_004, label="sigma 004", linestyle='--')
    plt.plot(results_006, label="sigma 006", linestyle='--')
    plt.plot(results_008, label="sigma 008", linestyle="--")
    plt.plot(results_01, label="sigma 01",linestyle="--")

    plt.title(metric)
    plt.xlabel('Photo ID')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def clip(img):
    img = np.minimum(np.ones(img.shape), img)
    img = np.maximum(np.zeros(img.shape), img)
    return img

def ksvd_denoise_image(noisy_filename, noisy_dir, n_components):
    start = time.time()
    print(f"--- Denoising image {noisy_filename} ... ---")
    split_sigma = noisy_filename.split(".")[0].split("_")[-1]
    img = img_as_float(imread(os.path.join(noisy_dir, noisy_filename)))
    patch_size = (5, 5)
    patches = image.extract_patches_2d(img, patch_size)
    signals = patches.reshape(patches.shape[0], -1)
    mean = np.mean(signals, axis=1)[:, np.newaxis]
    signals -= mean
    aksvd = ApproximateKSVD(n_components=n_components)
    dictionary = aksvd.fit(signals[:10000]).components_
    gamma = aksvd.transform(signals)
    reduced = gamma.dot(dictionary) + mean
    reduced_img = image.reconstruct_from_patches_2d(reduced.reshape(patches.shape), img.shape)
    io.imsave(f'denoisy_images/{split_sigma}/denoised_{noisy_filename}.JPG', clip(reduced_img))
    end = time.time()
    elapsed = end - start
    print(f"Successfully denoised and saved image denoised_{noisy_filename}.JPG in {round(elapsed, 2)} seconds")

def diff_image(original_filename, sig_denoi_dir, denoisy_filename):
    before = Image.open(os.path.join(ORG_IMAGE_DIR, original_filename))
    after = Image.open(os.path.join(sig_denoi_dir, denoisy_filename))

    before = np.asarray(before)
    after = np.asarray(after)

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    (score, diff) = ssim(before_gray, after_gray, full=True)
    print("Image similarity", score)

    diff = (diff * 255).astype("uint8")

    plt.figure(figsize=(10,10))
    plt.imshow(diff)
    plt.title(f"{denoisy_filename}")
    plt.show()