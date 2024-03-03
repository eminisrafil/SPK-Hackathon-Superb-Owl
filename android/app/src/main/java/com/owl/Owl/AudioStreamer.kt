package com.owl.Owl

import android.annotation.SuppressLint
import android.content.Context
import android.media.*
import android.net.Uri
import android.os.Looper
import androidx.media3.common.MediaItem
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.ConcatenatingMediaSource
import androidx.media3.exoplayer.source.ProgressiveMediaSource
import io.socket.client.IO
import io.socket.client.Socket
import java.net.URI
import java.util.logging.Handler
import kotlin.concurrent.thread
import kotlin.experimental.or
import android.content.Intent
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import org.json.JSONObject

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

class AudioStreamer(private val context: Context, private val captureUUID: String) {

    private var audioRecord: AudioRecord? = null
    private var audioEncoder: MediaCodec? = null
    private var socket: Socket? = null
    private val serverUrl = AppConstants.apiBaseURL
    private val sampleRate = 44100
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
    private val deviceName = "android"
    private var audioFocusChangeListener: AudioManager.OnAudioFocusChangeListener? = null
    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private var exoPlayer: ExoPlayer? = null
    private var concatenatingMediaSource = ConcatenatingMediaSource()
    @Volatile
    private var isStreaming = false

    init {
        initializeSocket()
        initializeExoPlayer()
    }

    private fun initializeSocket() {
        try {
            val options = IO.Options.builder()
                .setExtraHeaders(mapOf("Authorization" to listOf("Bearer ${AppConstants.clientToken}")))
                .build()
            socket = IO.socket(URI.create(serverUrl), options)
            socket?.on(Socket.EVENT_CONNECT) {
                println("Connected to socket")
            }?.on("audio_file") { args ->
                if (args.isNotEmpty() && args[0] is String) {
                    val mediaUrl = args[0] as String
                    println(mediaUrl)
                    addMediaToQueue("${AppConstants.apiBaseURL}/${AppConstants.assistantAudioPath}/${mediaUrl}")
                }
            }?.on("final_audio_file") { args ->
                val mainThreadHandler = android.os.Handler(Looper.getMainLooper())

                mainThreadHandler.post {
                    concatenatingMediaSource = ConcatenatingMediaSource()
                initializeExoPlayer()
                }
            }?.on("vibes") { args ->
                // Handle 'vibes' event
                println(args[0])

                if (args.isNotEmpty() && args[0] is JSONObject) {
                    val jsonObject = args[0] as JSONObject
                    val vibes = jsonObject.optInt("vibes") // Defaults to 0 if not found
                    val prompt = jsonObject.optString("prompt") // Defaults to empty string if not found
                    val vibesPercent = jsonObject.optInt("vibesPercent") // Defaults to 0 if not found

                    println(vibes)
                    println(prompt)
                    println(vibesPercent)

                    val intent = Intent(VIBES_EVENT).apply {
                        putExtra(VIBRATIONS_KEY, vibes)
                        putExtra(PROMPT_KEY, prompt)
                        putExtra(VIBES_PERCENT_KEY, vibesPercent)
                    }
                    LocalBroadcastManager.getInstance(context).sendBroadcast(intent)
                }


//                val args: Map<String, Any> = args
//
//// Updated check for a Map structure
//                if (args.isNotEmpty() && args[0] is Map<*, *>) {
//                    val argsMap = args[0] as Map<String, Any>
//                    // Now you can proceed with your existing check
//                    if (args.containsKey("vibes") && args["vibes"] is Int &&
//                        args.containsKey("prompt") && args["prompt"] is String &&
//                        args.containsKey("vibesPercent") && args["vibesPercent"] is Int) {
//                        // Successfully passed the check, now you can safely use these values
//                        val vibrations = args["vibes"] as Int
//                        val prompt = args["prompt"] as String
//                        val vibesPercent = args["vibesPercent"] as Int
//
//                        val intent = Intent(VIBES_EVENT).apply {
//                            putExtra(VIBRATIONS_KEY, vibrations)
//                            putExtra(PROMPT_KEY, prompt)
//                            putExtra(VIBES_PERCENT_KEY, vibesPercent) // Adding the vibesPercent to the intent
//                        }
//                        LocalBroadcastManager.getInstance(context).sendBroadcast(intent)
//                    }
//                }


//                if (args.size >= 3 && args[0] is Int && args[1] is String && args[2] is Int) {
//                    val vibrations = args[0] as Int
//                    val prompt = args[1] as String
//                    val vibesPercent = args[2] as Int // Extracting the vibesPercent value
//
//                    val intent = Intent(VIBES_EVENT).apply {
//                        putExtra(VIBRATIONS_KEY, vibrations)
//                        putExtra(PROMPT_KEY, prompt)
//                        putExtra(VIBES_PERCENT_KEY, vibesPercent) // Adding the vibesPercent to the intent
//                    }
//                    LocalBroadcastManager.getInstance(context).sendBroadcast(intent)
//                }

//                if (args.containsKey("vibes") && args.containsKey("prompt") && args.containsKey("vibesPercent")) {
//                    val vibrations = args["vibes"] as? Int ?: 0 // Providing a default value if null or not an Int
//                    val prompt = args["prompt"] as? String ?: "" // Providing a default value if null or not a String
//                    val vibesPercent = args["vibesPercent"] as? Int ?: 0 // Providing a default value if null or not an Int
//
//                    val intent = Intent(VIBES_EVENT).apply {
//                        putExtra(VIBRATIONS_KEY, vibrations)
//                        putExtra(PROMPT_KEY, prompt)
//                        putExtra(VIBES_PERCENT_KEY, vibesPercent)
//                    }
//                    LocalBroadcastManager.getInstance(context).sendBroadcast(intent)
//                }


//                if (args.size >= 2 && args[0] is Int && args[1] is String) {
//                    val vibrations = args[0] as Int
//                    val prompt = args[1] as String
//                    val intent = Intent(VIBES_EVENT).apply {
//                        putExtra(VIBRATIONS_KEY, vibrations)
//                        putExtra(PROMPT_KEY, prompt)
//                    }
//                    LocalBroadcastManager.getInstance(context).sendBroadcast(intent)
//                }
            }
            socket?.connect()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    companion object {
        const val VIBES_EVENT = "com.owl.Owl.VIBES_EVENT"
        const val VIBRATIONS_KEY = "vibrations"
        const val PROMPT_KEY = "prompt"
        const val VIBES_PERCENT_KEY  = "vibesPercent"
    }

    private fun initializeExoPlayer() {
        exoPlayer = ExoPlayer.Builder(context).build().apply {
            playWhenReady = true
        }
    }

    private fun addMediaToQueue(url: String) {
        val mainThreadHandler = android.os.Handler(Looper.getMainLooper())

        mainThreadHandler.post {
            val mediaItem = MediaItem.fromUri(Uri.parse(url))
            val dataSourceFactory = DefaultHttpDataSource.Factory()
            val mediaSource = ProgressiveMediaSource.Factory(dataSourceFactory).createMediaSource(mediaItem)

            concatenatingMediaSource.addMediaSource(mediaSource)

            if (exoPlayer?.playbackState == ExoPlayer.STATE_IDLE || exoPlayer?.playbackState == ExoPlayer.STATE_ENDED) {
                if (exoPlayer?.playbackState == ExoPlayer.STATE_IDLE) {
                    exoPlayer?.setMediaSource(concatenatingMediaSource)
                }
                exoPlayer?.prepare()
                exoPlayer?.play()
            } else if (exoPlayer?.playbackState == ExoPlayer.STATE_READY && !exoPlayer?.isPlaying!!) {
                exoPlayer?.play()
            }
        }
    }

    @android.annotation.SuppressLint("MissingPermission")
    fun startStreaming() {
        setupAudioRecord()
        setupAudioEncoder()

        audioRecord?.startRecording()
        audioEncoder?.start()
        isStreaming = true

        thread(start = true) { captureAndEncodeLoop() }
    }

    @SuppressLint("MissingPermission")
    private fun setupAudioRecord() {
        audioRecord = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.MIC)
            .setAudioFormat(AudioFormat.Builder()
                .setEncoding(audioFormat)
                .setSampleRate(sampleRate)
                .setChannelMask(channelConfig)
                .build())
            .setBufferSizeInBytes(bufferSize)
            .build()
    }

    private fun setupAudioEncoder() {
        val format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, sampleRate, 1)
        format.setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
        format.setInteger(MediaFormat.KEY_BIT_RATE, 64000)

        audioEncoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
        audioEncoder?.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
    }

    private fun captureAndEncodeLoop() {
        val inputBuffer = ByteArray(bufferSize)
        val bufferInfo = MediaCodec.BufferInfo()
        while (isStreaming) {
            val readResult = audioRecord?.read(inputBuffer, 0, inputBuffer.size) ?: 0
            if (readResult > 0) {
                encode(inputBuffer, readResult, bufferInfo)
            }
        }
        audioEncoder?.signalEndOfInputStream()
        releaseResources()
    }

    private fun encode(input: ByteArray, length: Int, bufferInfo: MediaCodec.BufferInfo) {
        val inputBufferIndex = audioEncoder?.dequeueInputBuffer(10000) ?: -1
        if (inputBufferIndex >= 0) {
            val inputBuffer = audioEncoder?.getInputBuffer(inputBufferIndex)
            inputBuffer?.clear()

            // Ensure we do not exceed the buffer's capacity
            val bytesToEncode = minOf(inputBuffer?.remaining() ?: 0, length)

            inputBuffer?.put(input, 0, bytesToEncode)
            audioEncoder?.queueInputBuffer(inputBufferIndex, 0, bytesToEncode, System.nanoTime() / 1000, 0)
        }

        var outputBufferIndex = audioEncoder?.dequeueOutputBuffer(bufferInfo, 10000) ?: -1
        while (outputBufferIndex >= 0) {
            val outputBuffer = audioEncoder?.getOutputBuffer(outputBufferIndex)
            val outData = ByteArray(bufferInfo.size)
            outputBuffer?.get(outData)
            outputBuffer?.clear()

            // Prepend ADTS header
            val adtsHeader = ByteArray(7)
            addADTSHeader(adtsHeader, bufferInfo.size + adtsHeader.size)

            // Combine ADTS header and encoded AAC frame
            val packet = ByteArray(adtsHeader.size + outData.size)
            System.arraycopy(adtsHeader, 0, packet, 0, adtsHeader.size)
            System.arraycopy(outData, 0, packet, adtsHeader.size, outData.size)

            // Send the packetized data
            socket?.emit("audio_data", packet, deviceName, captureUUID)

            audioEncoder?.releaseOutputBuffer(outputBufferIndex, false)
            outputBufferIndex = audioEncoder?.dequeueOutputBuffer(bufferInfo, 0) ?: -1
        }
    }


    private fun addADTSHeader(packet: ByteArray, packetLen: Int) {
        val profile = 2 // AAC LC
        val freqIdx = 4 // 44.1KHz
        val chanCfg = 1 // Mono

        // fill in ADTS data
        packet[0] = 0xFF.toByte()
        packet[1] = 0xF9.toByte()
        packet[2] = ((profile - 1) shl 6).toByte() or ((freqIdx shl 2).toByte()) or ((chanCfg shr 2).toByte())
        packet[3] = ((chanCfg and 3) shl 6).toByte() or ((packetLen shr 11).toByte())
        packet[4] = ((packetLen and 0x7FF) shr 3).toByte()
        packet[5] = ((packetLen and 7) shl 5).toByte() or 0x1F
        packet[6] = 0xFC.toByte()
    }

    private fun releaseResources() {
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null

        audioEncoder?.stop()
        audioEncoder?.release()
        audioEncoder = null

        socket?.disconnect()
    }

    fun stopStreaming() {
        isStreaming = false
    }
}

