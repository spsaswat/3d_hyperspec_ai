import numpy as np
import cv2
import spectral
import matplotlib.pyplot as plt

def normalize(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img

# Registration pipeline
def register_fx10_fx17(fx10_hdr,fx17_hdr):

    print("Loading cubes...")

    fx10 = spectral.open_image(fx10_hdr)
    fx17 = spectral.open_image(fx17_hdr)

    cube10 = fx10.load().astype(np.float32)
    cube17 = fx17.load().astype(np.float32)

    print("FX10:",cube10.shape)
    print("FX17:",cube17.shape)

    # Create grayscale
    wl10 = np.array(fx10.metadata["wavelength"], dtype=float)
    wl17 = np.array(fx17.metadata["wavelength"], dtype=float)

    idx10 = np.where((wl10 >= 900) & (wl10 <= 1000))[0]
    idx17 = np.where((wl17 >= 900) & (wl17 <= 1000))[0]
    print(idx10)
    print(idx17)

    I10 = normalize(np.mean(cube10[:, :, idx10], axis=2))
    I17 = normalize(np.mean(cube17[:, :, idx17], axis=2))
    H10,W10 = I10.shape

    I17 = cv2.resize(I17,(W10,H10))


    img1 = (I10*255).astype(np.uint8)
    img2 = (I17*255).astype(np.uint8)

    orb = cv2.ORB_create(4000)

    kp1,des1 = orb.detectAndCompute(img1,None)
    kp2,des2 = orb.detectAndCompute(img2,None)

    print("Keypoints FX10:",len(kp1))
    print("Keypoints FX17:",len(kp2))

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des1,des2,k=2)

    good=[]

    for m,n in matches:

        if m.distance < 0.75*n.distance:
            good.append(m)

    print("Good matches:",len(good))


    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    M,inliers = cv2.estimateAffine2D(
        pts2,
        pts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=5
    )

    print("Affine inliers:",np.sum(inliers))
    print("Warping cube...")

    bands = cube17.shape[2]

    cube17_resized = np.zeros((H10,W10,bands),dtype=np.float32)

    for b in range(bands):
        cube17_resized[:,:,b] = cv2.resize(
            cube17[:,:,b],
            (W10,H10)
        )


    cube17_registered = np.zeros_like(cube17_resized)

    for b in range(bands):

        cube17_registered[:,:,b] = cv2.warpAffine(
            cube17_resized[:,:,b],
            M,
            (W10,H10)
        )

    warped = cv2.warpAffine(I17,M,(W10,H10))

    overlay = np.zeros((H10,W10,3))

    overlay[:,:,1] = normalize(I10)
    overlay[:,:,0] = normalize(warped)

    overlay = np.clip(overlay,0,1)


    plt.figure(figsize=(12,6))

    plt.subplot(151)
    plt.title("FX10")
    plt.imshow(I10,cmap='gray')

    plt.subplot(152)
    plt.title("FX17")
    plt.imshow(fx17[:, :, 0], cmap='gray')

    plt.subplot(153)
    plt.title("Resized FX17")
    plt.imshow(I17,cmap='gray')

    plt.subplot(154)
    plt.title("Warped FX17")
    plt.imshow(warped,cmap='gray')

    plt.subplot(155)
    plt.title("Overlay")
    plt.imshow(overlay)

    plt.tight_layout()
    plt.show()

    return cube17_registered,M,inliers

if __name__ == "__main__":

    fx10_hdr = "/home/sriram/ANU/data/no-background/004-specim-fx10_no_background.hdr"
    fx17_corr_hdr = "/home/sriram/ANU/data/no-background/corrected/004-specim-fx17_no_background_corrected-FX17.hdr"
    register_fx10_fx17(fx10_hdr,fx17_corr_hdr)