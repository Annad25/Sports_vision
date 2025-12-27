# Volleyball Player, Ball & Referee Tracking with 2D Tactical Map

## Overview

This project implements a complete computer vision pipeline for volleyball broadcast videos. It detects, tracks, and classifies players, the referee, and the ball, and projects all entities onto a synchronized top-down Hawk-Eye–style 2D tactical map.

The system is designed to be **spatially robust and explainable**, avoiding jersey-color or appearance-based rules.

---

## Sport Chosen

## Volleyball

---

## Pipeline Summary

---

Video
 → Manual court corner selection (once)
 → Optical flow corner tracking
 → YOLOv8 detection
 → ByteTrack multi-object tracking
 → Role & team classification (spatial logic)
 → Homography projection
 → 2D tactical map overlay

---

## Key Components

### Court Detection & Mapping

* Court corners are manually selected once when the full court is visible
* Corners are tracked across frames using **Lucas–Kanade optical flow**
* A fixed homography maps the broadcast view to a top-down court
* This avoids unreliable automatic court detection on broadcast footage

---

### Detection & Tracking

* **YOLOv8 (COCO)** for player, referee, and ball detection
* **ByteTrack** for multi-object tracking
* Stable tracking IDs maintained across frames

---

### Ball Tracking

The volleyball is small and frequently missed by detectors.

A custom temporal tracker:

* Stores recent ball positions
* Estimates velocity
* Predicts position when detections are missing

Predicted ball positions are visually marked in the output.

---

### Team & Role Classification (No Color Heuristics)

Jersey color is intentionally **not used** because:

* Libero wears different colors
* Referee clothing(colors) can match players
* Lighting varies across footage

Instead, **geometry-based rules** are applied:

* **Players** → Assigned to left or right team based on court side
* **Referee** → Near the net line but outside active play area
* **Spectators / non-players** → Outside buffered court boundaries

This significantly reduces common misclassification issues.

---

### 2D Tactical Map

* Players, referee, and ball are projected using homography
* Court lines, net, and attack zones are drawn
* Map is synchronized frame-by-frame with the broadcast video

---

## Output

* Annotated broadcast video
* Embedded Hawk-Eye–style 2D tactical map
* Output shared via **Google Drive** (as required)

---

## Assumptions for this solution

* Camera is mostly static (broadcast-style view)
* Full court is visible at least once for initialization
* Net orientation is roughly vertical in the frame
* Single-camera video input

---

## Known Limitations

* Manual court initialization required
* Homography is static (no dynamic recalibration yet)
* Referee may be misclassified as spectator if very close to play area
* Spectators may still be detected as persons but filtered spatially
* No appearance-based team clustering by design

---
---

## Improvements

* Automatic court detection
* Dynamic homography updates
* Kalman filter for smoother ball motion
* Optional appearance embeddings for team refinement

