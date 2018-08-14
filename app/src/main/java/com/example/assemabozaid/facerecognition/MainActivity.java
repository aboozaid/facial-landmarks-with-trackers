package com.example.assemabozaid.facerecognition;

import android.content.Context;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.face.Face;
import org.opencv.face.Facemark;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.tracking.MultiTracker;
import org.opencv.tracking.TrackerMedianFlow;
import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;

import static org.opencv.core.Core.FILLED;
import static org.opencv.objdetect.Objdetect.CASCADE_SCALE_IMAGE;

public class MainActivity extends AppCompatActivity implements JavaCameraView.CvCameraViewListener2 {
    static String TAG = MainActivity.class.getName();
    private JavaCameraView cameraView;
    private CascadeClassifier cascadeClassifier;
    private int rotation;
    private MultiTracker tracker;
    private Mat mRgba, mGray;
    private MatOfRect faces;
    private ArrayList<Rect2d> points;
    private boolean loaded = false;
    private Facemark marks;
    private AsyncTasks.loadModules loader;
    private ArrayList<MatOfPoint2f> landmarks;
    private MatOfRect2d trackerPoints;
    private Timer myTimer = new Timer(true);
    private boolean isBackCamera = false;
    private BaseLoaderCallback callbackLoader = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch(status) {
                case BaseLoaderCallback.SUCCESS:
                    loader = new AsyncTasks.loadModules(getApplicationContext(), "haarcascade_frontalface_alt2.xml", onLoadedEnd);
                    loader.execute();
                    loader = new AsyncTasks.loadModules(getApplicationContext(), "lbfmodel.yaml", onLoadedEnd);
                    loader.execute();
                    faces = new MatOfRect();
                    cameraView.enableView();
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
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (JavaCameraView)findViewById(R.id.camera_view);
        cameraView.setVisibility(SurfaceView.VISIBLE);
        cameraView.setCvCameraViewListener(this);

        Button flip = (Button)findViewById(R.id.flip_button);
        flip.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                cameraView.disableView();
                if(cameraView.getCameraIndex() == CameraBridgeViewBase.CAMERA_ID_FRONT) {
                    cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
                    isBackCamera = true;
                }
                else {
                    cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
                    isBackCamera = false;
                }
                cameraView.enableView();
            }
        });
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraView != null)
            cameraView.disableView();
    }
    @Override
    protected void onDestroy(){
        super.onDestroy();
        if(cameraView != null)
            cameraView.disableView();
    }
    @Override
    protected void onResume(){
        super.onResume();
        if(OpenCVLoader.initDebug()) {
            Log.i(TAG, "System Library Loaded Successfully");
            callbackLoader.onManagerConnected(BaseLoaderCallback.SUCCESS);
        } else {
            Log.i(TAG, "Unable To Load System Library");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, callbackLoader);
        }
    }
    @Override
    public void onCameraViewStarted(int width, int height) {
        rotation = ((WindowManager) getApplicationContext().getSystemService(Context.WINDOW_SERVICE)).getDefaultDisplay().getRotation();
        cameraView.initResolutions();
        mGray = new Mat();
        mRgba = new Mat();

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();

        applyOrientation(mRgba, isBackCamera, rotation); // change false to true when you're using the back camera
        Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_RGB2GRAY);


        if(tracker != null && tracker.update(mGray, trackerPoints)) {
            MatOfRect faces = new MatOfRect();
            Rect2d[] facePoints = trackerPoints.toArray();
            ArrayList<Rect> rec = new ArrayList<>();
            for(int i=0; i<facePoints.length; i++)
            {
                Imgproc.rectangle(mRgba, facePoints[i].tl(), facePoints[i].br(), new Scalar(255, 255, 255), 1);
                rec.add(new Rect(facePoints[i].tl(), facePoints[i].br()));
            }
            if(loaded) {
                faces.fromList(rec);
                landmarks = new ArrayList<>();
                if(marks.fit(mGray, faces, landmarks)) drawLandmarks();
            }
        } else {
            myTimer.scheduleAtFixedRate(new detection(), 1000, 5 * 1000);
        }

        return mRgba;
    }
    private void drawLandmarks() {
        for (int i=0; i<landmarks.size(); i++) {
            MatOfPoint2f lm = landmarks.get(i);
            Point[] points = lm.toArray();

            if(points.length == 68) {
                drawPolyLines(0,16, lm); // Jaw line
                drawPolyLines(17,21, lm); // Left eyebrow
                drawPolyLines(22,26, lm); // Right eyebrow
                drawPolyLines(27,30, lm); // Nose bridge
                drawPolyLines(30,35, lm); // Lower nose
                drawPolyLines(36, 41, lm); // Left eye
                drawPolyLines(42, 47, lm); // Right Eye
                drawPolyLines(48, 59, lm); // Outer lip
                drawPolyLines(60, 67, lm); // Inner lip
            } else {
                for(Point p: points) {
                    Imgproc.circle(mRgba, p, 3, new Scalar(219, 13, 37), FILLED);
                }
            }
        }
    }
    private void drawPolyLines(final int start, final int end, MatOfPoint2f marks) {
        ArrayList<MatOfPoint> polylinesPoints = new ArrayList<>();
        for(int i= start; i <= end; i++) {
            polylinesPoints.add(new MatOfPoint(new Point(marks.toArray()[i].x, marks.toArray()[i].y)));
        }
        Imgproc.polylines(mRgba,polylinesPoints,true,new Scalar(72, 188, 200),2, Imgproc.LINE_8, 0);
    }
    private void applyOrientation(Mat rgba, boolean clockwise, int rotation) {
        if (rotation == Surface.ROTATION_0) {
            // Rotate clockwise / counter clockwise 90 degrees
            Mat rgbaT = rgba.t();
            Core.flip(rgbaT, rgba, clockwise ? 1 : -1);
            rgbaT.release();
        } else if (rotation == Surface.ROTATION_270) {
            // Rotate clockwise / counter clockwise 180 degrees
            Mat rgbaT = rgba.t();
            Core.flip(rgba.t(), rgba, clockwise ? 1 : -1);
            rgbaT.release();
            Mat rgbaT2 = rgba.t();
            Core.flip(rgba.t(), rgba, clockwise ? 1 : -1);
            rgbaT2.release();
        }
    }

    private class detection extends TimerTask {

        @Override
        public void run() {
            cascadeClassifier.detectMultiScale(mGray, faces, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, new Size(30, 30));
            if (!faces.empty()) {
                Rect[] facesArray = faces.toArray();
                Rect2d[] trackerArr = new Rect2d[facesArray.length];
                points = new ArrayList<>();
                trackerPoints = new MatOfRect2d();
                tracker = MultiTracker.create();
                for (int i = 0; i < facesArray.length; i++) {
                    points.add(new Rect2d(facesArray[i].tl(), facesArray[i].br()));
                    // there is a bunch of collection of trackers you can isntead of MedianFlow
                    // https://docs.opencv.org/3.1.0/d2/d0a/tutorial_introduction_to_tracker.html
                    tracker.add(TrackerMedianFlow.create(), mGray, points.get(i));
                    trackerArr[i] = points.get(i);
                }
                trackerPoints.fromArray(trackerArr);
            }
        }
    }
    private AsyncTasks.loadModules.Callback onLoadedEnd = new AsyncTasks.loadModules.Callback() {
        @Override
        public void onLoaded(String path) {
            if(path.contains("alt2"))
                cascadeClassifier = new CascadeClassifier(path);
            else {
                marks = Face.createFacemarkLBF();
                marks.loadModel(path);
                loaded = true;
            }
        }
    };
}
