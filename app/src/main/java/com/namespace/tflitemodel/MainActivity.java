package com.namespace.tflitemodel;

import androidx.appcompat.app.AppCompatActivity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    ImageView imageView;
    TextView textView;
    String result = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);
    }
   public void predict(View view){

        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.hen);

       ImageProcessor imageProcessor =  new ImageProcessor.Builder()
                                        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                                        .build();

       TensorImage tImage = new TensorImage(DataType.UINT8);

       tImage = imageProcessor.process(TensorImage.fromBitmap(bitmap));

       TensorBuffer probabilityBuffer = TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);

       try{

           MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(this,"mobilenet_v1_1.0_224_quant.tflite");

           Interpreter tflite = new Interpreter(tfliteModel);

           // Running inference
           tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer());

       } catch (IOException e){
           Log.e("tfliteSupport", "Error reading model", e);
       }

       final String MOBILE_NET_LABELS = "labels.txt";

       List<String> mobilenetlabels = null;

       try {
           mobilenetlabels = FileUtil.loadLabels(this, MOBILE_NET_LABELS);
       } catch (IOException e) {
           Log.e("tfliteSupport", "Error reading label file", e);
       }

// Post-processor which dequantize the result
       TensorProcessor probabilityProcessor = new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

       if (null != mobilenetlabels) {
           // Map of labels and their corresponding probability
           TensorLabel labels = new TensorLabel(mobilenetlabels, probabilityProcessor.process(probabilityBuffer));
           // Create a map to access the result based on label
           Map<String, Float> resultsMap = labels.getMapWithFloatValue();

           for (String key : resultsMap.keySet()) {
               Float value = resultsMap.get(key);
               if (value >= 0.50){
                   String roundOff = String.format("%.2f", value);
                   result = key + " " + roundOff;
               }
               Log.i( "Info",  key + " " + value);
           }
           textView.append(result);
       }
    }
}
