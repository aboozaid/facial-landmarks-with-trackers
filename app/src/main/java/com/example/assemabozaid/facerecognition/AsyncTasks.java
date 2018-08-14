package com.example.assemabozaid.facerecognition;

import android.content.Context;
import android.os.AsyncTask;
import android.util.Log;

import org.opencv.face.Face;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

/**
 * Created by Assem Abozaid on 8/14/2018.
 */

public class AsyncTasks {
    static String TAG = AsyncTasks.class.getSimpleName();

    static class loadModules extends AsyncTask<Void, Void, String> {
        private Context context;
        private Callback callback;
        private String file;
        interface Callback {
            void onLoaded(String path);
        }
        public loadModules(Context context,String fileName ,Callback callback) {
            this.context = context;
            this.callback = callback;
            this.file = fileName;
        }
        @Override
        protected String doInBackground(Void... strings) {
            try {
                // Copy the resource into a temp file so OpenCV can load it
                InputStream is = null;
                if(file.contains("alt2"))
                    is = context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
                else
                    is = context.getResources().openRawResource(R.raw.lbfmodel);
                File modelDir = context.getDir("files", Context.MODE_PRIVATE);
                File modelFile = new File(modelDir, file);
                FileOutputStream os = new FileOutputStream(modelFile);


                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
                is.close();
                os.close();

                Log.i(TAG, "Loaded the file success");
                return modelFile.getAbsolutePath();
            } catch (Exception e) {
                return null;
            }
        }
        @Override
        protected void onPostExecute(String result) {
            if(result != null)
                callback.onLoaded(result);
            else
                Log.i(TAG, "Failed to load the module");
        }
    }
}
