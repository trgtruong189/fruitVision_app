import 'dart:io';

import 'package:get/get.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

import '../P.dart';
import '../models/History.dart';

class ClassificationController extends GetxController {
  List<String> label = [
    'Bean',
    'Bitter_Gourd',
    'Bottle_Gourd',
    'Brinjal',
    'Broccoli',
    'Cabbage',
    'Capsicum',
    'Carrot',
    'Cauliflower',
    'Cucumber',
    'Papaya',
    'Potato',
    'Pumpkin',
    'Radish',
    'Tomato',
  ];
  late Interpreter _interpreter;
  RxString result = "No classification yet".obs;

  @override
  void onInit() {
    // TODO: implement onInit
    super.onInit();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      print("Đang tải mô hình từ assets...");
      final options = InterpreterOptions();
      _interpreter = await Interpreter.fromAsset(
        'assets/tflite/model_4.tflite',
        options: options,
      );
      print("Mô hình local đã tải thành công!");
    } catch (e) {
      print("Lỗi tải mô hình local: $e");
    }
    print("Input tensor shape: ${_interpreter.getInputTensor(0).shape}");
    print("Output tensor shape: ${_interpreter.getOutputTensor(0).shape}");
  }

  Future<void> classifyImage(File imageFile) async {
    img.Image? image = img.decodeImage(imageFile.readAsBytesSync());
    if (image == null) {
      result.value = "Error loading image";
      return;
    }

    image = img.copyResize(image, width: 224, height: 224);
    List input = List.generate(
      1,
          (i) => List.generate(
        224,
            (y) => List.generate(224, (x) => List.filled(3, 0.0)),
      ),
    );

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        var pixel = image.getPixelSafe(x, y);
        if (pixel is img.PixelUint8) {
          num red = pixel.r;
          num green = pixel.g;
          num blue = pixel.b;

          input[0][y][x][0] = red.toDouble() /255.0;
          input[0][y][x][1] = green.toDouble() / 255.0;
          input[0][y][x][2] = blue.toDouble() / 255.0;
        }
      }
    }
    final stopwatch = Stopwatch()..start();
    List output = List.generate(1, (i) => List.filled(15, 0.0));
    _interpreter.run(input, output);
    stopwatch.stop();
    print("Classification time: ${stopwatch.elapsedMilliseconds} ms");
    List<double> probabilities = output[0].cast<double>();
    int labelIndex = probabilities.indexOf(
      probabilities.reduce((a, b) => a > b ? a : b),
    );
    result.value = label[labelIndex];
  }

  Future<void> postHistory(History his) async {
    P.fireStore.collection("History").add(his.toJson());
  }

  void clearResult() {
    result.value = "No classification yet";
  }
}


