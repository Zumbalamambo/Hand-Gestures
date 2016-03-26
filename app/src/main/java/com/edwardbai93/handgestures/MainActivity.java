package com.edwardbai93.handgestures;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private ProjectView mOpenCvCameraView;
    Hand hand = new Hand();
    Frame frame = new Frame();

    public static final int SAMPLE_BACK = -1; // sample average color of background area
    public static final int SAMPLE_MODE = 0; // sample average color of the hand
    public static final int DETECTION_MODE = 1; // generates binary image for review
    public static final int TRACKING_MODE = 2; // draw contour and recognize gesture
    private int mode = SAMPLE_BACK; // set initial mode

    // Load libraries
    static {
        System.loadLibrary("MyLib");
        System.loadLibrary("opencv_java");
    }

    // Establish connection with OpenCV Manager
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i("Hand Gesture", "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        frame.initFrame();

        mOpenCvCameraView = (ProjectView) findViewById(R.id.main_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        frame.startFrame(width, height);
    }

    public void onCameraViewStopped() {
        frame.releaseFrame();
    }

    public void switchMode(View view) {
        if (mode == SAMPLE_BACK) mode = SAMPLE_MODE;
        else if (mode == SAMPLE_MODE) mode = DETECTION_MODE;
        else if (mode == DETECTION_MODE) mode = TRACKING_MODE;
        else mode = DETECTION_MODE;
    }

    public void resample(View view) {
        mode = SAMPLE_BACK;
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        frame.readInputFrame(inputFrame);

        if (mode == SAMPLE_BACK) {
            frame.sampleBackgroundColor();
            return frame.getRGBAFrame();
        } else if (mode == SAMPLE_MODE) {
            // Samples the average colors of the hand
            frame.sampleHandColor();
            return frame.getRGBAFrame();
        } else if (mode == DETECTION_MODE) {
            // Generates binary image of the hand, whose area is colored white
            frame.produceBinaryImage();
            return frame.getMaskFrame();
        } else if (mode == TRACKING_MODE) {
            // Tracks hand and draws contours on the frame
            frame.produceBinaryImage();
            frame.draw(hand);
            return frame.getRGBAFrame();
        }
        else return frame.getRGBAFrame();
    }

    /*@Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }*/

}
