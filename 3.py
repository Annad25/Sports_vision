import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np

# Input video path
VIDEO_PATH = "Video4.mp4"


class CourtDetector:
    """
    Handles:
    1. Manual selection of volleyball court corners (once)
    2. Tracking those corners across frames using optical flow
    """

    def __init__(self):
        # Stores the original selected court corners
        self.court_corners = None

        # Previous grayscale frame 
        self.prev_frame_gray = None

        # Corner positions from the previous frame
        self.prev_corners = None

        # Lucas–Kanade Optical Flow parameters
        self.lk_params = dict(
            winSize=(21, 21),   # Size of search window
            maxLevel=3,         # Number of pyramid levels
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            )
        )

    def manual_court_selection(self, frame):
        """
        Allows user to manually click the 4 court corners in order:
        Top-Left → Top-Right → Bottom-Right → Bottom-Left
        """
        print("\nSelect 4 corners: TL → TR → BR → BL")

        corners = []
        clone = frame.copy()

        # Mouse callback for selecting points
        def click_event(event, x, y, flags, param):
            nonlocal corners, clone

            # Register left mouse clicks
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append([x, y])

                # Draw selected point
                cv2.circle(clone, (x, y), 8, (0, 255, 0), -1)

                # Label the point number
                cv2.putText(
                    clone,
                    str(len(corners)),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # Draw line between consecutive points
                if len(corners) > 1:
                    cv2.line(
                        clone,
                        tuple(corners[-2]),
                        tuple(corners[-1]),
                        (0, 255, 0),
                        3
                    )

                # Close the polygon after 4 points
                if len(corners) == 4:
                    cv2.line(
                        clone,
                        tuple(corners[-1]),
                        tuple(corners[0]),
                        (0, 255, 0),
                        3
                    )
                    cv2.putText(
                        clone,
                        "Press ENTER to confirm",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                cv2.imshow("Court Selection", clone)

        # Setup selection window
        cv2.namedWindow("Court Selection", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Court Selection", click_event)
        cv2.imshow("Court Selection", clone)

        # Wait until user confirms or cancels
        while True:
            key = cv2.waitKey(1) & 0xFF

            # ENTER confirms selection (only if 4 points selected)
            if key == 13 and len(corners) == 4:
                break

            # ESC cancels selection
            if key == 27:
                cv2.destroyWindow("Court Selection")
                return None

        cv2.destroyWindow("Court Selection")

        corners = np.array(corners, dtype=np.float32)
        print("Selected corners:", corners.tolist())

        # Return in shape expected by calcOpticalFlowPyrLK
        return corners.reshape(-1, 1, 2)

    def initialize(self, frame):
        """
        Initializes corner tracking by:
        - Manual selection
        - Saving initial grayscale frame
        """
        corners = self.manual_court_selection(frame)

        if corners is not None:
            self.court_corners = corners
            self.prev_corners = corners.copy()
            self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True

        return False

    def track_corners(self, frame):
        """
        Tracks previously selected corners using optical flow
        """
        if self.prev_corners is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical flow estimation
        new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame_gray,
            gray,
            self.prev_corners,
            None,
            **self.lk_params
        )

        # Update stored grayscale frame
        self.prev_frame_gray = gray

        # If tracking succeeded for all corners
        if new_corners is not None and status is not None and np.all(status == 1):
            self.prev_corners = new_corners
            return new_corners.reshape(-1, 2)

        # Fallback: return last known corner positions
        return self.prev_corners.reshape(-1, 2)


def main():
    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    detector = CourtDetector()

    print("Video playing")
    print("Press 's' when FULL court is visible")

    # --- Initialization phase ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended before selection")
            cap.release()
            return

        display = frame.copy()
        cv2.putText(
            display,
            "Waiting for full court... Press 's'",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow("Initialization", display)
        key = cv2.waitKey(30) & 0xFF

        # Start manual corner selection
        if key == ord('s'):
            if detector.initialize(frame):
                break

        # Exit completely
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Initialization")

    # --- Tracking phase ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        corners = detector.track_corners(frame)

        # Draw tracked corners
        if corners is not None:
            for c in corners:
                cv2.circle(frame, tuple(c.astype(int)), 6, (255, 0, 0), -1)

        cv2.imshow("Corner Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
