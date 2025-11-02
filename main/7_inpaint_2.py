import os
import cv2
import numpy as np
import time
from tqdm import tqdm

class ManualInpainter:
    def __init__(self, patch_size=9, stride=10):
        self.patch_size = patch_size
        self.stride = stride
        self.circle_mask = self.create_circular_mask(patch_size)

    def create_circular_mask(self, size):
        """Create a circular mask to apply on the patch."""
        y, x = np.ogrid[:size, :size]
        center = (size - 1) / 2
        mask = (x - center)**2 + (y - center)**2 <= (center)**2
        return mask.astype(np.uint8)[..., np.newaxis]  # shape (size, size, 1)

    def inpaint(self, image, mask):
        result = image.copy()
        work_mask = mask.copy()

        while np.any(work_mask > 0):
            points = np.column_stack(np.where(work_mask > 0))
            if len(points) == 0:
                break

            point = points[0]
            patch = self.get_patch(result, point)

            best_patch = self.find_best_patch(result, work_mask, tuple(point))

            if best_patch is not None:
                ph = self.patch_size
                y, x = point
                y1, x1 = best_patch
                try:
                    src_patch = result[y1 - ph//2:y1 + ph//2 + 1, x1 - ph//2:x1 + ph//2 + 1]
                    if src_patch.shape != (ph, ph, 3):
                        continue
                    dst_slice = result[y - ph//2:y + ph//2 + 1, x - ph//2:x + ph//2 + 1]
                    np.copyto(dst_slice, src_patch, where=self.circle_mask.astype(bool))
                    result[y - ph//2:y + ph//2 + 1, x - ph//2:x + ph//2 + 1] = dst_slice
                    work_mask[y - ph//2:y + ph//2 + 1, x - ph//2:x + ph//2 + 1] *= (1 - self.circle_mask[:, :, 0])
                except:
                    break
            else:
                break

        return result

    def get_patch(self, image, point):
        y, x = point
        ph = self.patch_size
        return image[y - ph//2:y + ph//2 + 1, x - ph//2:x + ph//2 + 1]

    def find_best_patch(self, image, mask, point):
        h, w = image.shape[:2]
        ph = self.patch_size
        best_score = float('inf')
        best_patch = None

        patch = self.get_patch(image, point)
        if patch.shape != (ph, ph, 3):
            return None

        for i in range(ph//2, h - ph//2, self.stride):
            for j in range(ph//2, w - ph//2, self.stride):
                if np.any(mask[i - ph//2:i + ph//2 + 1, j - ph//2:j + ph//2 + 1]):
                    continue

                candidate = image[i - ph//2:i + ph//2 + 1, j - ph//2:j + ph//2 + 1]
                if candidate.shape != patch.shape:
                    continue

                score = np.sum(((candidate - patch) * self.circle_mask)**2)

                if score < best_score:
                    best_score = score
                    best_patch = (i, j)

        return best_patch


def inpaint_image(img_path, mask_path, output_path, inpainter):
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    if image is None or mask is None:
        print(f"❌ Error loading image or mask: {img_path}, {mask_path}")
        return

    start_time = time.time()
    result = inpainter.inpaint(image, mask)
    duration = time.time() - start_time

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"✅ Done: {os.path.basename(img_path)} | Time: {duration:.2f}s")


def process_folder(folder_type, data_dir, masks_dir, output_dir):
    print(f"\n🔧 Inpainting {folder_type} images...")
    folder_path = os.path.join(data_dir, folder_type)
    mask_path = os.path.join(masks_dir, folder_type)
    output_path = os.path.join(output_dir, folder_type)

    inpainter = ManualInpainter(patch_size=11, stride=5)  # Smaller stride = better quality

    for case_folder in tqdm(os.listdir(folder_path), desc=folder_type):
        case_input = os.path.join(folder_path, case_folder)
        case_mask = os.path.join(mask_path, case_folder)
        case_output = os.path.join(output_path, case_folder)

        if not os.path.isdir(case_input):
            continue

        for fname in os.listdir(case_input):
            if fname.endswith(('.png', '.jpg', '.jpeg')):
                input_img_path = os.path.join(case_input, fname)
                mask_img_path = os.path.join(case_mask, fname)
                output_img_path = os.path.join(case_output, fname)

                inpaint_image(input_img_path, mask_img_path, output_img_path, inpainter)


def main():
    data_dir = "C:/Users/NISHANK/Desktop/miniproject/data"
    masks_dir = "C:/Users/NISHANK/Desktop/miniproject/UNET_image/masks"
    output_dir = "C:/Users/NISHANK/Desktop/miniproject/inpainted"

    for folder in ["train", "test", "val"]:
        process_folder(folder, data_dir, masks_dir, output_dir)


if __name__ == "__main__":
    main()
