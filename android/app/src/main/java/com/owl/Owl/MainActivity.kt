package com.owl.Owl

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.UUID

import com.airbnb.lottie.LottieAnimationView
import com.airbnb.lottie.LottieDrawable

import android.os.Handler
import android.os.Looper
import kotlin.random.Random
import android.widget.TextView
import android.animation.Animator

import androidx.localbroadcastmanager.content.LocalBroadcastManager


import android.os.VibrationEffect
import android.os.Vibrator

import android.content.BroadcastReceiver
import android.content.IntentFilter

import android.content.Context


class ForegroundService : Service() {
    private val CHANNEL_ID = "owl_foreground_service"

    override fun onCreate() {
        super.onCreate()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channelName = CHANNEL_ID
            val channel = NotificationChannel(CHANNEL_ID, channelName, NotificationManager.IMPORTANCE_DEFAULT)
            val manager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }

    fun updateLabel() {
        //R.layout.activity_main.mytext
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d("ForegroundService", "Service is starting...")
        val notificationBuilder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            Notification.Builder(this) // For older devices
        }

        val notification = notificationBuilder
            .setContentTitle("App is running in background")
            .setContentText("Doing something.")
            .setSmallIcon(android.R.drawable.ic_dialog_info) // Changed for testing
            .build()

        startForeground(1, notification)
        Log.d("ForegroundService", "Foreground service started with notification.")

        // Your background task here
        return START_STICKY
    }

    override fun onBind(intent: Intent): IBinder? {
        return null
    }
}

class MainActivity : AppCompatActivity() {

    private lateinit var cameraHandler: CameraHandler
    private lateinit var audioStreamer: AudioStreamer
    private val TAG = "CaptureActivity"
    private val REQUEST_PERMISSIONS = 101
    private val captureUUID: String = UUID.randomUUID().toString().replace("-", "")

    private lateinit var emojiTextView: TextView
    private val emojis = arrayOf("😀", "😁", "😂", "🤣", "😃", "😄", "😅", "😆", "😉", "😊") // Define your set of emojis here
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var lottieAnimationView: LottieAnimationView
    private val updateEmojiRunnable = object : Runnable {
        override fun run() {
//            updateEmoji()
//            handler.postDelayed(this, 3000) // Schedule the next execution in 3 seconds (3000ms)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraHandler = CameraHandler(this, captureUUID)
        startService(Intent(this, ForegroundService::class.java))
        requestPermissions()

        emojiTextView = findViewById(R.id.mytext) // Make sure this ID matches your TextView's ID in the layout
        startUpdatingEmojis()

        lottieAnimationView = findViewById(R.id.lottieAnimationView)
        startLottieAnimation()

        startPeriodicBroadcast()

        lottieAnimationView.addAnimatorListener(object : Animator.AnimatorListener {
            override fun onAnimationStart(animation: Animator) {
                Log.d("LottieAnimation", "Animation started")
            }

            override fun onAnimationEnd(animation: Animator) {
                Log.d("LottieAnimation", "Animation ended")
            }

            override fun onAnimationCancel(animation: Animator) {
                Log.d("LottieAnimation", "Animation canceled")
            }

            override fun onAnimationRepeat(animation: Animator) {
                Log.d("LottieAnimation", "Animation repeated")
            }
        })

        LocalBroadcastManager.getInstance(this).registerReceiver(
            vibesReceiver, IntentFilter(AudioStreamer.VIBES_EVENT)
        )
    }

    private val vibesReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == AudioStreamer.VIBES_EVENT) {
                val vibrations = intent.getIntExtra(AudioStreamer.VIBRATIONS_KEY, 0)
                val prompt = intent.getStringExtra(AudioStreamer.PROMPT_KEY) ?: "Default Prompt"
                val positivity = intent.getIntExtra(AudioStreamer.VIBES_PERCENT_KEY, 50) // Default to a neutral value if not provided
                //updateEmojiLabel(prompt)
                updateEmojiLabel(positivity.toString()) // Change here
                vibrateDevice(vibrations)
                //playAppropriateLottieAnimation(positivity)
            }
        }

        private fun playAppropriateLottieAnimation(positivity: Int) {
            when {
                positivity < 40 -> lottieAnimationView.setAnimation("RedVibe.json")
                positivity <= 60 -> lottieAnimationView.setAnimation("ChillVibe.json")
                positivity > 60 -> lottieAnimationView.setAnimation("testLottie.json")
            }
            lottieAnimationView.playAnimation()
        }
    }

    private fun startPeriodicBroadcast() {
//        handler.postDelayed(object : Runnable {
//            override fun run() {
//                val testIntent = Intent(AudioStreamer.VIBES_EVENT)
//                val positivity = Random.nextInt(100) // Generates a random positivity score from 0 to 99
//                testIntent.putExtra("positivity", positivity)
//
//                LocalBroadcastManager.getInstance(this@MainActivity).sendBroadcast(testIntent)
//
//                handler.postDelayed(this, 5000) // Reschedule the next broadcast in 5 seconds
//            }
//        }, 5000) // Initial delay
    }

    private fun vibrateDevice(vibrations: Int) {
        val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        if (vibrator.hasVibrator()) { // Check if the device has a vibrator
            val milliseconds = vibrations.toLong() // Convert vibrations to milliseconds (as needed)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                // New API
                vibrator.vibrate(VibrationEffect.createOneShot(milliseconds, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                // Deprecated in API 26
                vibrator.vibrate(milliseconds)
            }
        }
    }

    private fun updateEmojiLabel(text: String) {
        emojiTextView.text = text
    }

    private fun startLottieAnimation() {
        lottieAnimationView.playAnimation()
    }

    private fun pauseLottieAnimation() {
        if (lottieAnimationView.isAnimating) {
            lottieAnimationView.pauseAnimation()
        }
    }

    private fun resumeLottieAnimation() {
        if (!lottieAnimationView.isAnimating) {
            lottieAnimationView.resumeAnimation()
        }
    }

    private fun stopLottieAnimation() {
        lottieAnimationView.cancelAnimation()
    }

    // Call this function to make the animation loop indefinitely
    private fun loopLottieAnimation() {
        lottieAnimationView.repeatCount = LottieDrawable.INFINITE
    }

    private fun startUpdatingEmojis() {
        handler.post(updateEmojiRunnable) // Start the periodic update
    }

    private fun updateEmoji() {
        val randomIndex = Random.nextInt(emojis.size) // Get a random index for the emoji array
        emojiTextView.text = emojis[randomIndex] // Update the TextView with a new emoji
    }

    private fun requestPermissions() {
        val requiredPermissions = arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA)
        val permissionsToRequest = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }.toTypedArray()

        if (permissionsToRequest.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, permissionsToRequest, REQUEST_PERMISSIONS)
        } else {
            permissionsGranted()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_PERMISSIONS && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
            permissionsGranted()
        } else {
            Log.e(TAG, "Permissions not granted by the user.")
        }
    }

    private fun permissionsGranted() {
        cameraHandler.startBackgroundThread()
        cameraHandler.openCamera()
        audioStreamer = AudioStreamer(this, captureUUID)
        audioStreamer.startStreaming()
    }

    override fun onResume() {
        super.onResume()
        cameraHandler.startBackgroundThread()
    }

    override fun onPause() {
        cameraHandler.stopBackgroundThread()
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraHandler.closeCamera()
        cameraHandler.stopBackgroundThread()
    }
}
