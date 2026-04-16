import cv2
import json

image_path = "backend/data/golden_pcb.jpg"
output_path = "backend/data/roi_map.json"

img = cv2.imread(image_path)
clone = img.copy()

drawing = False
x_start, y_start = -1, -1

roi_map = {}
class_map = {
    "0": "R",  # resistor
    "1": "C",  # capacitor
    "2": "IC",
    "3": "D",  # diode
    "4": "U"   # other
}

class_counts = {k: 1 for k in class_map.values()}


def draw_box(event, x, y, flags, param):
    global x_start, y_start, drawing, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = clone.copy()
            cv2.rectangle(img, (x_start, y_start), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        x1, y1 = x_start, y_start
        x2, y2 = x, y

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        print("\nSelect class:")
        print("0: Resistor | 1: Capacitor | 2: IC | 3: Diode | 4: Other")
        cls = input("Enter class number: ")

        if cls not in class_map:
            print("Invalid class, skipping")
            return

        prefix = class_map[cls]
        name = f"{prefix}{class_counts[prefix]}"
        class_counts[prefix] += 1

        roi_map[name] = {
            "box": [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)],
            "class": int(cls)
        }

        print(f"Saved ROI: {name}")


cv2.namedWindow("ROI Annotator")
cv2.setMouseCallback("ROI Annotator", draw_box)

print("Instructions:")
print("- Drag mouse to draw box")
print("- Enter class when prompted")
print("- Press 's' to save, 'q' to quit")

while True:
    cv2.imshow("ROI Annotator", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        with open(output_path, "w") as f:
            json.dump(roi_map, f, indent=4)
        print("ROI map saved!")

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
