import numpy as np
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class ImageProcessig:
    @staticmethod
    def decode_predictions(scores, geometry, min_confidence):
        (num_rows, num_cols) = scores.shape[2:4]
        rectangles = []
        confidences = []
        for y in range(0, num_rows):
            scores_data = scores[0, 0, y]
            x_data0 = geometry[0, 0, y]
            x_data1 = geometry[0, 1, y]
            x_data2 = geometry[0, 2, y]
            x_data3 = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]
            for x in range(0, num_cols):
                if scores_data[x] < min_confidence:
                    continue
                (offset_x, offset_y) = (x * 4.0, y * 4.0)
                angle = angles_data[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = x_data0[x] + x_data2[x]
                w = x_data1[x] + x_data3[x]
                end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
                end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)
                rectangles.append((start_x, start_y, end_x, end_y))
                confidences.append(scores_data[x])
        return rectangles, confidences

    @staticmethod
    def loop_over_boxes_get_text(boxes, ratio, original, padding, orig):
        results = []
        orig_width = original[0]
        orig_height = original[1]
        ratio_width = ratio[0]
        ratio_height = ratio[1]
        for (start_x, start_y, end_x, end_y) in boxes:
            start_x = int(start_x * ratio_width)
            start_y = int(start_y * ratio_height)
            end_x = int(end_x * ratio_width)
            end_y = int(end_y * ratio_height)
            diff_abscissa = int((end_x - start_x) * padding)
            diff_ordinate = int((end_y - start_y) * padding)
            start_x = max(0, start_x - diff_abscissa)
            start_y = max(0, start_y - diff_ordinate)
            end_x = min(orig_width, end_x + (diff_abscissa * 2))
            roi = orig[start_y:end_y, start_x:end_x]
            end_y = min(orig_height, end_y + (diff_ordinate * 2))
            config = ("-l eng --oem 1 --psm 3")
            text = pytesseract.image_to_string(roi, config=config)
            results.append(((start_x, start_y, end_x, end_y), text))
        results = sorted(results, key=lambda r: r[0][1])
        return results

    @staticmethod
    def display(results, orig, speech):
        for ((start_x, start_y, end_x, end_y), text) in results:
            print("{}\n".format(text))
            print(start_x, start_y, end_x, end_y)
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            text.capitalize()
            if speech in text:
                output = orig.copy()
                cv2.rectangle(output, (start_x, start_y), (end_x, end_y),
                              (0, 0, 255), 2)
                cv2.putText(output, text, (start_x, end_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("Text Detection", output)
                cv2.waitKey(0)
