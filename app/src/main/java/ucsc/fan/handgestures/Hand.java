package ucsc.fan.handgestures;

import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.contourArea;


public class Hand {
    // Data structures for approximating contours
    public List<MatOfPoint> contours = new ArrayList<>();
    public int contourMaxId = -1;
    public Mat hierachy = new Mat();
    public MatOfPoint2f approx_contour = new MatOfPoint2f();

    // Data structures for calculating convex hull
    public List<MatOfPoint> hull_point = new ArrayList<>();
    public MatOfInt hullx = new MatOfInt();

    // Data structures for calculating convexity defects
    public MatOfInt4 defects = new MatOfInt4();
    public List<Point> defect_points = new ArrayList<>();

    // Bounding rectangle of the hand
    public Rect bounding_rect;

    public void findBiggestContour() {
        /**
         * Mutate contourMaxId of the hand object
         * Select largest contour based on contour area
         * and number of contour points
         */
        int idx = -1;
        int cNum = 0;
        double max_area = 0.0;
        for (int i = 0; i < contours.size(); i++) {
            int curNum = contours.get(i).toList().size();
            double m = contourArea(contours.get(i));
            if (m > max_area || curNum > cNum) {
                idx = i;
                max_area =  Math.max(max_area,m);
                cNum = Math.max(cNum, curNum);
            }
        }
        contourMaxId = idx;
    }

    public boolean isHand(Mat img) {
        /**
         * Detect if the contour area encloses a hand by
         * the size and position of the bounding rectangle
         */
        if (bounding_rect == null) return false;
        int centerX = bounding_rect.x + bounding_rect.width/2;
        int centerY = bounding_rect.y + bounding_rect.height/2;

        if (contourMaxId == -1) // cannot select a largest contour
            return false;
        else if (bounding_rect.height < img.rows() / 2 || bounding_rect.width < img.cols() / 4) // area too small
            return false;
        else if ((centerX < img.cols() / 4) || (centerX > img.cols() * 3 / 4) ||
                centerY < img.rows() / 4 || centerY > img.rows() * 3 / 4) // center of rectangle is too far from the middle
            return false;
        else
            return true;
    }

    public double distanceP2P(Point a, Point b){
        /**
         * Obtain the Euclidean distance between two points
         */
        return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
    }

    public double xOffset(Point a, Point b) {
        return Math.abs(a.x - b.x);
    }

    public double yOffset(Point a, Point b) {
        return Math.abs(a.y - b.y);
    }

    public double getAngle(Point start, Point furthest, Point end){
        /**
         * Obtain the angle in degrees between the two lines joining
         * start-furthest and end-furthest respectively
         */
        double l1 = distanceP2P(furthest, start);
        double l2 = distanceP2P(furthest, end);
        double dot = (start.x - furthest.x) * (end.x - furthest.x) +
                (start.y - furthest.y) * (end.y - furthest.y);
        return Math.acos(dot / (l1 * l2)) * 180 / Math.PI;
    }
}
