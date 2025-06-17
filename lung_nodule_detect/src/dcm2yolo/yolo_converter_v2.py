import numpy as np
import cv2  # For saving images


class YOLOConverterV2:
    def points_to_yolo_bbox(self, points, original_img_shape, target_img_shape=(512, 512)):
        """
        Converts a list of (x, y) pixel coordinates to a YOLO bounding box.
        original_img_shape: (rows, cols) of the image where points are defined.
        target_img_shape: (height, width) of the final YOLO image.
        Returns (class_id, center_x_norm, center_y_norm, width_norm, height_norm)
        """
        if not points:
            return None

        points_arr = np.array(points)
        min_x = np.min(points_arr[:, 0])
        max_x = np.max(points_arr[:, 0])
        min_y = np.min(points_arr[:, 1])
        max_y = np.max(points_arr[:, 1])

        # Ensure valid box
        if max_x <= min_x or max_y <= min_y:
            return None

        original_rows, original_cols = original_img_shape
        target_height, target_width = target_img_shape  # YOLO usually expects (width, height) for target

        # Box dimensions in original pixel coordinates
        box_width_orig = max_x - min_x
        box_height_orig = max_y - min_y
        center_x_orig = min_x + box_width_orig / 2
        center_y_orig = min_y + box_height_orig / 2

        # Normalize to target image dimensions
        # The points are in pixel coordinates of the *original* DICOM image.
        # We need to scale these to the *target* YOLO image size.

        # YOLO format: <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
        # All normalized by image width and height.

        # Normalize coordinates by original image dimensions
        x_center_norm = center_x_orig / original_cols
        y_center_norm = center_y_orig / original_rows
        width_norm = box_width_orig / original_cols
        height_norm = box_height_orig / original_rows

        # Class ID for 'nodule' is 0
        class_id = 0

        # Clamp values to be within [0, 1] and ensure width/height are positive
        x_center_norm = np.clip(x_center_norm, 0.0, 1.0)
        y_center_norm = np.clip(y_center_norm, 0.0, 1.0)
        width_norm = np.clip(width_norm, 0.001, 1.0)  # min width to avoid zero
        height_norm = np.clip(height_norm, 0.001, 1.0)  # min height to avoid zero

        return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

# Example Usage:
# if __name__ == '__main__':
#     converter = YOLOConverterV2()
#     # Example points from your 068.xml for Nodule 001
#     # (1207,545), (1204,551), (1221,544), (1210,542)
#     example_points = [[1207,545], [1204,551], [1221,544], [1210,542]]
#     # Assume original DICOM was 2022x2022 as per your earlier log
#     original_shape_example = (2022, 2022)
#     target_shape_example = (512, 512)

#     yolo_str = converter.points_to_yolo_bbox(example_points, original_shape_example, target_shape_example)
#     print(f"YOLO string: {yolo_str}")