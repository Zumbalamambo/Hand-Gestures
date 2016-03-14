package ucsc.fan.handgestures;

/**
 * This is the main class involving binary image acquisition and processing
 * Credit of multipoint color sampling and background subtraction goes to 
 * Yalun Qin's app (https://github.com/eaglesky/HandGestureApp)
 * 
 * Part of Yalun's idea/code is referenced:
 * 1. Line 320, 344: Core.add(imgOut, sampleMats[i], imgOut);
 * 2. Line 351-364: produceBinaryImage()
 */

import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.FONT_HERSHEY_SIMPLEX;
import static org.opencv.core.Core.bitwise_and;
import static org.opencv.core.Core.circle;
import static org.opencv.core.Core.inRange;
import static org.opencv.core.Core.putText;
import static org.opencv.core.Core.rectangle;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_NONE;
import static org.opencv.imgproc.Imgproc.GaussianBlur;
import static org.opencv.imgproc.Imgproc.MORPH_CLOSE;
import static org.opencv.imgproc.Imgproc.MORPH_RECT;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;
import static org.opencv.imgproc.Imgproc.approxPolyDP;
import static org.opencv.imgproc.Imgproc.boundingRect;
import static org.opencv.imgproc.Imgproc.convexHull;
import static org.opencv.imgproc.Imgproc.convexityDefects;
import static org.opencv.imgproc.Imgproc.drawContours;
import static org.opencv.imgproc.Imgproc.findContours;
import static org.opencv.imgproc.Imgproc.getStructuringElement;
import static org.opencv.imgproc.Imgproc.medianBlur;
import static org.opencv.imgproc.Imgproc.morphologyEx;

public class Frame {
    private Mat mRgba, mIntermediateMat, mFGMask; // frames to be utilized
    private static final int SAMPLE_NUM = 8; // number of sample points in sample mode
    private Point[][] samplePoints;
    private double[][] avgHandColor;
    private double[][] avgBackgroundColor;
    private Scalar lowerBound = new Scalar(0, 0, 0);
    private Scalar upperBound = new Scalar(0, 0, 0);
    private Mat[] sampleMats;

    // Constant threshold values of average color
    private double[] handColorLowerRadius = new double[3];
    private double[] handColorUpperRadius = new double[3];
    private double[] handColorBackLowerRadius = new double[3];
    private double[] handColorBackUpperRadius = new double[3];
    private static final int COLOR_SPACE = Imgproc.COLOR_RGB2Lab;

    public void initFrame() {
        /**
         * Called during onCreate() method to
         * 1. create new instances for various data structures
         * 2. initialize constants
         */
        samplePoints = new Point[SAMPLE_NUM][2];
        for (int i = 0; i < SAMPLE_NUM; i++) {
            for (int j = 0; j < 2; j++) {
                samplePoints[i][j] = new Point();
            }
        }

        avgHandColor = new double[SAMPLE_NUM][3];
        avgBackgroundColor = new double[SAMPLE_NUM][3];

        if (sampleMats == null) {
            sampleMats = new Mat[SAMPLE_NUM];
            for (int i = 0; i < SAMPLE_NUM; i++)
                sampleMats[i] = new Mat();
        }

        initCLowerUpper(40, 40, 10, 10, 10, 10);
        initCBackLowerUpper(50, 50, 3, 3, 3, 3);
    }

    public void startFrame(int width, int height) {
        /**
         * Called during onCameraViewStarted() method to
         * initialize three important frames
         */
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mFGMask = new Mat(height,width,CvType.CV_8UC1);
    }

    public void releaseFrame() {
        /**
         * Releases matrices during onCameraViewStopped()
         */
        mRgba.release();
        mIntermediateMat.release();
        mFGMask.release();
    }

    public Mat getRGBAFrame() {
        /**
         * Returns current color (RGBA) frame
         */
        return mRgba;
    }

    public Mat getMaskFrame() {
        /**
         * Returns current binary (mask) frame
         */
        return mFGMask;
    }

    public void readInputFrame(CvCameraViewFrame inputFrame) {
        /**
         * Read input frame and pre-process the colors
         */
        mRgba = inputFrame.rgba();
        GaussianBlur(mRgba, mRgba, new Size(5, 5), 5, 5);
        Imgproc.cvtColor(mRgba, mIntermediateMat, COLOR_SPACE);
    }

    public void draw(Hand hand) {
        /**
         * ORIGINAL CONTENT
         * Draws largest contour, convex hull and convexity defects
         * Count number of fingertips and defects
         */
        hand.contours.clear();
        hand.defect_points.clear();

        medianBlur(mFGMask, mFGMask, 5);
        findContours(mFGMask, hand.contours, hand.hierachy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        hand.findBiggestContour();
        if (hand.contourMaxId != -1) {
            hand.approx_contour.fromList(hand.contours.get(hand.contourMaxId).toList());
            approxPolyDP(hand.approx_contour, hand.approx_contour, 1, false);
            hand.contours.get(hand.contourMaxId).fromList(hand.approx_contour.toList());


            hand.bounding_rect = boundingRect(hand.contours.get(hand.contourMaxId));

            convexHull(hand.contours.get(hand.contourMaxId),hand.hullx,false);
            hand.hull_point.clear();
            for (int i = 0; i < hand.contours.size(); i++)
                hand.hull_point.add(new MatOfPoint());

            int[] contourIdx = hand.hullx.toArray();
            List<Point> tmp = new ArrayList<>();
            Point[] contourPts = hand.contours.get(hand.contourMaxId).toArray();

            for (int i = 0; i < contourIdx.length; i++) {
                tmp.add(contourPts[contourIdx[i]]);
            }

            // hand.hull_point.get(hand.contourMaxId) returns the locations of the points in the convex hull of the hand
            hand.hull_point.get(hand.contourMaxId).fromList(tmp);
            tmp.clear();

            if (contourPts.length >= 5 && contourIdx.length >= 5) {
                convexityDefects(hand.contours.get(hand.contourMaxId), hand.hullx, hand.defects);
                int[] defect_list = hand.defects.toArray();
                Point[] data = hand.contours.get(hand.contourMaxId).toArray();
                for (int i = 0; i < defect_list.length; i = i + 4) {
                    Point start = data[defect_list[i]];
                    Point end = data[defect_list[i + 1]];
                    Point furthest = data[defect_list[i + 2]];
                    if (hand.distanceP2P(end, furthest) > hand.bounding_rect.height / 5 &&
                            hand.distanceP2P(start, furthest) > hand.bounding_rect.height / 5 &&
                            hand.getAngle(start, furthest,end) < 80){
                        hand.defect_points.add(furthest);
                    }
                }

                List<Point> ori_hull_p = hand.hull_point.get(hand.contourMaxId).toList();
                List<Point> new_hull_p = new ArrayList<>();
                for (Point p:ori_hull_p) new_hull_p.add(p);
                int size = ori_hull_p.size();
                for (int i = 1; i < size ;) {
                    Point prev = new_hull_p.get(i==0 ? size - 1:i-1);
                    Point curr = new_hull_p.get(i);
                    Point next = new_hull_p.get(i==size-1?0:i + 1);
                    if (hand.distanceP2P(prev, curr) < hand.bounding_rect.height / 10 ||
                            hand.distanceP2P(curr, next) < hand.bounding_rect.height / 10 ||
                            hand.xOffset(prev, curr) < hand.bounding_rect.width / 10 ||
                            hand.xOffset(curr, next) < hand.bounding_rect.width / 10 ||
                            hand.getAngle(prev, curr, next) > 165) {
                        new_hull_p.remove(i);
                        size--;
                    }
                    else i++;
                }

                for (int i = 0;i<size;) {
                    Point p = new_hull_p.get(i);
                    if (p.y > mRgba.rows() - hand.bounding_rect.height / 4) {
                        new_hull_p.remove(p);
                        size--;
                    }
                    else i++;
                }

                int defects = hand.defect_points.size();
                int fingers = new_hull_p.size();
                String result = new String();

                switch (defects) {
                    case 4: result = "5"; break;
                    case 3: result = "4"; break;
                    case 2: result = "3"; break;
                    case 1: result = "2";break;
                    case 0:
                        if (fingers ==1) result = "1";
                        else result = "0";
                        break;
                    default: break;
                }
                putText(mRgba, result, new Point(500,100), FONT_HERSHEY_SIMPLEX, 4,new Scalar(108,230,94),2);
            }
        }

        if (hand.isHand(mRgba)) {
            rectangle(mRgba, hand.bounding_rect.tl(), hand.bounding_rect.br(), new Scalar(0, 0, 255), 3);
            drawContours(mRgba, hand.hull_point, hand.contourMaxId, new Scalar(255, 0, 0), 2);
            drawContours(mRgba, hand.contours, hand.contourMaxId, new Scalar(0, 255, 0), 3);
            for (Point defect : hand.defect_points) circle(mRgba, defect, 5, new Scalar(255, 255, 0), 2);
        }
        /////////////////////////////////////
    }

    public void sampleHandColor() {
        /**
         * Samples and stores hand colors to an avgBackgroundColor matrix,
         * which contains 3 channel color sampled by 8 squares
         */
        int cols = mRgba.cols();
        int rows = mRgba.rows();
        int squareLen = rows / 20;

        samplePoints[0][0] = new Point(cols * 7 / 45, rows * 7 / 27);
        samplePoints[1][0] = new Point(cols * 12 / 45, rows * 5 / 36);
        samplePoints[2][0] = new Point(cols * 25 / 72, rows / 9);
        samplePoints[3][0] = new Point(cols * 4 / 9, rows / 6);
        samplePoints[4][0] = new Point(cols * 7 / 12, rows / 2);
        samplePoints[5][0] = new Point(cols / 4, rows * 37 / 45);
        samplePoints[6][0] = new Point(cols * 3 / 8, rows * 37 / 45);
        samplePoints[7][0] = new Point(cols * 295 / 720, rows * 322 / 540);

        for (int i = 0; i < SAMPLE_NUM; i++) {
            samplePoints[i][1].x = samplePoints[i][0].x + squareLen;
            samplePoints[i][1].y = samplePoints[i][0].y + squareLen;
        }

        for (int i = 0; i < SAMPLE_NUM; i++) {
            Core.rectangle(mRgba, samplePoints[i][0], samplePoints[i][1], new Scalar(255,0,0),5);
        }
        for (int i = 0; i < SAMPLE_NUM; i++) {
            for (int j = 0; j < 3; j++) {
                avgHandColor[i][j] = (mIntermediateMat.get((int)(samplePoints[i][0].y+squareLen/2), (int)(samplePoints[i][0].x+squareLen/2)))[j];
            }
        }
    }

    public void sampleBackgroundColor() {
        /**
         * Samples and stores background colors to an avgBackgroundColor matrix,
         * which contains 3 channel color sampled by 8 squares
         */
        int cols = mFGMask.cols();
        int rows = mFGMask.rows();
        int squareLen = rows/20;

        samplePoints[0][0] = new Point(cols / 6, rows / 3);
        samplePoints[1][0] = new Point(cols / 6, rows * 2 / 3);
        samplePoints[2][0] = new Point(cols / 2, rows / 6);
        samplePoints[3][0] = new Point(cols / 3, rows / 2);
        samplePoints[4][0] = new Point(cols * 2 / 3, rows / 2);
        samplePoints[5][0] = new Point(cols / 2, rows * 5 / 6);
        samplePoints[6][0] = new Point(cols * 5 / 6, rows / 3);
        samplePoints[7][0] = new Point(cols * 5 / 6, rows * 2 / 3);

        for (int i = 0; i < SAMPLE_NUM; i++) {
            samplePoints[i][1].x = samplePoints[i][0].x + squareLen;
            samplePoints[i][1].y = samplePoints[i][0].y + squareLen;
            Core.rectangle(mRgba, samplePoints[i][0], samplePoints[i][1], new Scalar(0,0,255), 2);
        }

        for (int i = 0; i < SAMPLE_NUM; i++) {
            for (int j = 0; j < 3; j++) {
                avgBackgroundColor[i][j] = (mIntermediateMat.get((int)(samplePoints[i][0].y + squareLen/2), (int)(samplePoints[i][0].x + squareLen/2)))[j];
            }
        }
    }

    private void produceBinaryHandImage(Mat imgIn, Mat imgOut) {
        /**
         * Generates binary image thresholded only by sampled hand colors
         */
        for (int i = 0; i < SAMPLE_NUM; i++) {
            lowerBound.set(new double[]{avgHandColor[i][0]- handColorLowerRadius[0], avgHandColor[i][1]- handColorLowerRadius[1],
                    avgHandColor[i][2]- handColorLowerRadius[2]});
            upperBound.set(new double[]{avgHandColor[i][0]+ handColorUpperRadius[0], avgHandColor[i][1]+ handColorUpperRadius[1],
                    avgHandColor[i][2]+ handColorUpperRadius[2]});
            inRange(imgIn, lowerBound, upperBound, sampleMats[i]);
        }

        imgOut.release();
        sampleMats[0].copyTo(imgOut);

        for (int i = 1; i < SAMPLE_NUM; i++) {
            Core.add(imgOut, sampleMats[i], imgOut);
        }

        Imgproc.medianBlur(imgOut, imgOut, 3);
        morphologyEx(imgOut, imgOut, MORPH_CLOSE, getStructuringElement(MORPH_RECT, new Size(7, 7)), new Point(-1, -1), 1);
    }

    private void produceBinaryBackgroundImage(Mat imgIn, Mat imgOut) {
        /**
         * Generates binary image thresholded only by sampled background colors
         */
        for (int i = 0; i < SAMPLE_NUM; i++) {
            lowerBound.set(new double[]{avgBackgroundColor[i][0]- handColorBackLowerRadius[0], avgBackgroundColor[i][1]- handColorBackLowerRadius[1],
                    avgBackgroundColor[i][2]- handColorBackLowerRadius[2]});
            upperBound.set(new double[]{avgBackgroundColor[i][0]+ handColorBackUpperRadius[0], avgBackgroundColor[i][1]+ handColorBackUpperRadius[1],
                    avgBackgroundColor[i][2]+ handColorBackUpperRadius[2]});

            Core.inRange(imgIn, lowerBound, upperBound, sampleMats[i]);
        }

        imgOut.release();
        sampleMats[0].copyTo(imgOut);

        for (int i = 1; i < SAMPLE_NUM; i++) {
            Core.add(imgOut, sampleMats[i], imgOut);
        }

        Core.bitwise_not(imgOut, imgOut);
        Imgproc.medianBlur(imgOut, imgOut, 7);
    }

    public void produceBinaryImage() {
        /**
         * Generates the overall binary image for display and detection
         */
        Mat binTmpMat = new Mat();
        Mat binTmpMat2 = new Mat();
        handleBounds();
        produceBinaryHandImage(mIntermediateMat, binTmpMat);
        produceBinaryBackgroundImage(mIntermediateMat, binTmpMat2);

        bitwise_and(binTmpMat, binTmpMat2, binTmpMat);

        binTmpMat.copyTo(mFGMask);
    }

    private void handleBounds() {
        /**
         * Prevents color value from exceeding range 0 ~ 255
         */
        for (int i = 0; i < SAMPLE_NUM; i++) {
            for (int j = 0; j < 3; j++) {
                if (avgHandColor[i][j] < handColorLowerRadius[j])
                    handColorLowerRadius[j] = avgHandColor[i][j];
                if (avgHandColor[i][j] + handColorUpperRadius[j] > 255)
                    handColorUpperRadius[j] = 255 - avgHandColor[i][j];

                if (avgBackgroundColor[i][j] < handColorBackLowerRadius[j])
                    handColorBackLowerRadius[j] = avgBackgroundColor[i][j];
                if (avgBackgroundColor[i][j] + handColorBackUpperRadius[j] > 255)
                    handColorBackUpperRadius[j] = 255 - avgBackgroundColor[i][j];
            }
        }
    }

    private void initCLowerUpper(double cl1, double cu1, double cl2, double cu2, double cl3,
                                 double cu3) {
        /**
         * Initialize color threshold of the hand sample
         */
        handColorLowerRadius[0] = cl1;
        handColorUpperRadius[0] = cu1;
        handColorLowerRadius[1] = cl2;
        handColorUpperRadius[1] = cu2;
        handColorLowerRadius[2] = cl3;
        handColorUpperRadius[2] = cu3;
    }

    private void initCBackLowerUpper(double cl1, double cu1, double cl2, double cu2, double cl3,
                                     double cu3) {
        /**
         * Initialize color threshold of the background sample
         */
        handColorBackLowerRadius[0] = cl1;
        handColorBackUpperRadius[0] = cu1;
        handColorBackLowerRadius[1] = cl2;
        handColorBackUpperRadius[1] = cu2;
        handColorBackLowerRadius[2] = cl3;
        handColorBackUpperRadius[2] = cu3;
    }

}

