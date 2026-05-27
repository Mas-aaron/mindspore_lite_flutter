package com.agrichain.mindspore_lite_flutter

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import com.mindspore.lite.LiteSession
import com.mindspore.lite.MSTensor
import com.mindspore.lite.config.MSConfig
import com.mindspore.lite.config.DeviceType
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * MindsporeLiteFlutterPlugin
 *
 * Flutter plugin for on-device AI inference using Huawei MindSpore Lite 1.7.0.
 *
 * MethodChannel: "com.agrichain.mindspore"
 *
 * Supported methods:
 *   initModel(modelPath: String)          → Boolean
 *   predictImage(imagePath: String)       → Map<String, Any>
 *   disposeModel()                        → Boolean
 */
class MindsporeLiteFlutterPlugin : FlutterPlugin, MethodCallHandler {

    private lateinit var channel: MethodChannel
    private lateinit var appContext: Context

    // MindSpore Lite session (one per loaded model)
    private var session: LiteSession? = null
    private var modelPath: String? = null

    // Standard input size for AgriChain models (224×224)
    // Override by passing inputSize in initModel if needed
    private var inputSize: Int = 224

    companion object {
        private const val CHANNEL = "com.agrichain.mindspore"
        private const val TAG = "MindsporeLitePlugin"
        private const val PIXEL_SIZE = 3          // RGB
        private const val IMAGE_MEAN = 127.5f
        private const val IMAGE_STD = 127.5f
    }

    // ── FlutterPlugin lifecycle ───────────────────────────────

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        appContext = binding.applicationContext
        channel = MethodChannel(binding.binaryMessenger, CHANNEL)
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        closeSession()
    }

    // ── MethodChannel handler ─────────────────────────────────

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "initModel" -> {
                val assetPath = call.argument<String>("modelPath")
                    ?: return result.error("INVALID_ARGS", "modelPath is required", null)
                val size = call.argument<Int>("inputSize") ?: 224
                inputSize = size
                result.success(initModel(assetPath))
            }

            "predictImage" -> {
                val imagePath = call.argument<String>("imagePath")
                    ?: return result.error("INVALID_ARGS", "imagePath is required", null)
                if (session == null) {
                    return result.error("NOT_INITIALIZED", "Call initModel first", null)
                }
                try {
                    val output = runInference(imagePath)
                    result.success(output)
                } catch (e: Exception) {
                    result.error("INFERENCE_ERROR", e.message, null)
                }
            }

            "disposeModel" -> {
                closeSession()
                result.success(true)
            }

            "getPlatformVersion" -> {
                result.success("Android ${android.os.Build.VERSION.RELEASE}")
            }

            else -> result.notImplemented()
        }
    }

    // ── Model initialisation ──────────────────────────────────

    private fun initModel(assetModelPath: String): Boolean {
        return try {
            // Copy model from Flutter assets to internal storage
            val modelFile = copyAssetToFile(assetModelPath)
            if (!modelFile.exists()) {
                android.util.Log.e(TAG, "Model file not found after copy: ${modelFile.absolutePath}")
                return false
            }

            // Close any existing session
            closeSession()

            // Configure MindSpore Lite
            val config = MSConfig()
            config.init(DeviceType.DT_CPU, 2) // 2 threads

            // Create and build session
            val newSession = LiteSession.createSession(config)
            if (newSession == null) {
                android.util.Log.e(TAG, "Failed to create LiteSession")
                return false
            }

            val compiled = newSession.loadModelFromFile(modelFile.absolutePath)
            if (!compiled) {
                android.util.Log.e(TAG, "Failed to load model from: ${modelFile.absolutePath}")
                newSession.free()
                return false
            }

            session = newSession
            modelPath = modelFile.absolutePath
            android.util.Log.i(TAG, "Model loaded: $assetModelPath → ${modelFile.absolutePath}")
            true
        } catch (e: Exception) {
            android.util.Log.e(TAG, "initModel error: ${e.message}")
            false
        }
    }

    // ── Inference ─────────────────────────────────────────────

    private fun runInference(imagePath: String): Map<String, Any> {
        val sess = session ?: throw IllegalStateException("Session not initialized")

        val startMs = System.currentTimeMillis()

        // Load and preprocess image
        val bitmap = loadAndResizeBitmap(imagePath, inputSize)
        val inputBuffer = bitmapToByteBuffer(bitmap, inputSize)

        // Get input tensor and fill it
        val inputs = sess.inputs
        if (inputs.isNullOrEmpty()) {
            throw RuntimeException("Model has no inputs")
        }
        val inputTensor: MSTensor = inputs[0]
        inputTensor.setData(inputBuffer)

        // Run inference
        val success = sess.runGraph()
        if (!success) {
            throw RuntimeException("runGraph() failed")
        }

        // Read output tensor
        val outputs = sess.outputs
        if (outputs.isNullOrEmpty()) {
            throw RuntimeException("Model has no outputs")
        }
        val outputTensor: MSTensor = outputs.values.first()
        val outputData = outputTensor.floatData

        val inferenceMs = System.currentTimeMillis() - startMs

        return mapOf(
            "predictions" to outputData.toList(),
            "inferenceMs" to inferenceMs.toInt(),
        )
    }

    // ── Image preprocessing ───────────────────────────────────

    private fun loadAndResizeBitmap(imagePath: String, size: Int): Bitmap {
        val options = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val original = BitmapFactory.decodeFile(imagePath, options)
            ?: throw RuntimeException("Cannot decode image: $imagePath")

        return if (original.width == size && original.height == size) {
            original
        } else {
            val matrix = Matrix()
            val scaleX = size.toFloat() / original.width
            val scaleY = size.toFloat() / original.height
            matrix.setScale(scaleX, scaleY)
            val resized = Bitmap.createBitmap(original, 0, 0, original.width, original.height, matrix, true)
            if (resized != original) original.recycle()
            resized
        }
    }

    /**
     * Convert bitmap to float32 ByteBuffer normalised to [-1, 1].
     * Layout: NHWC (batch=1, height, width, channels=3)
     */
    private fun bitmapToByteBuffer(bitmap: Bitmap, size: Int): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * size * size * PIXEL_SIZE)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(size * size)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr 8) and 0xFF).toFloat()
            val b = (pixel and 0xFF).toFloat()
            byteBuffer.putFloat((r - IMAGE_MEAN) / IMAGE_STD)
            byteBuffer.putFloat((g - IMAGE_MEAN) / IMAGE_STD)
            byteBuffer.putFloat((b - IMAGE_MEAN) / IMAGE_STD)
        }

        byteBuffer.rewind()
        return byteBuffer
    }

    // ── Asset helper ──────────────────────────────────────────

    /**
     * Copy a Flutter asset to the app's internal files directory.
     * Returns the File object pointing to the copied file.
     * Skips copy if file already exists (cached).
     */
    private fun copyAssetToFile(assetPath: String): File {
        val fileName = assetPath.substringAfterLast("/")
        val outFile = File(appContext.filesDir, "mindspore_models/$fileName")

        if (!outFile.exists()) {
            outFile.parentFile?.mkdirs()
            appContext.assets.open(assetPath).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
            android.util.Log.i(TAG, "Copied asset $assetPath → ${outFile.absolutePath}")
        }

        return outFile
    }

    // ── Cleanup ───────────────────────────────────────────────

    private fun closeSession() {
        session?.free()
        session = null
        modelPath = null
    }
}
