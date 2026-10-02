package com.nickzam.server

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.nickzam.server.download.DownloadRepository
import com.nickzam.server.download.DownloadStatus
import com.nickzam.server.download.rememberModelDirSnapshot
import com.nickzam.server.inference.Sha256
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.security.SecureRandom

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Foreground-service notifications require an explicit grant on API 33+.
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        }
        setContent {
            NickZamTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    NickZamScreen()
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun NickZamScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val status by ServerState.status.collectAsState()
    val boundUrl by ServerState.boundUrl.collectAsState()
    val lastError by ServerState.lastError.collectAsState()
    val snapshot by rememberModelDirSnapshot(active = true)

    var selectedId by remember { mutableStateOf(Settings.selectedModelId(context)) }
    val model = findModelInfo(selectedId) ?: DEFAULT_MODEL
    val modelFile = remember(snapshot, model) {
        File(context.getExternalFilesDir(null), model.filename)
    }
    val installed = model.filename in snapshot.existingModels

    var portText by remember { mutableStateOf(Settings.port(context).toString()) }
    var apiKeyText by remember { mutableStateOf(Settings.apiKey(context)) }
    var bindLan by remember { mutableStateOf(Settings.bindLan(context)) }
    var tailscaleOn by remember { mutableStateOf(Settings.tailscale(context)) }
    val accessLabel by ServerState.accessLabel.collectAsState()
    var busy by remember { mutableStateOf<String?>(null) }
    var verifyNote by remember { mutableStateOf<String?>(null) }
    var engineLoaded by remember { mutableStateOf(false) }

    // Generation settings: local mirrors of Settings.* so the fields edit
    // smoothly and persist on change. Sampler values take effect on the next
    // request (server reads them whenever a request omits the param); the
    // context window is engine-baked, so changing it evicts engines to force a
    // rebuild at the new KV budget.
    var tempText by remember { mutableStateOf(Settings.temperature(context).toString()) }
    var topKText by remember { mutableStateOf(Settings.topK(context).toString()) }
    var topPText by remember { mutableStateOf(Settings.topP(context).toString()) }
    var ctxTokens by remember { mutableStateOf(Settings.contextTokens(context)) }
    val activeDownloads = remember { mutableStateMapOf<String, Long>() }
    var downloadState by remember { mutableStateOf<DownloadStatus?>(null) }
    val downloadRepo = remember { DownloadRepository(context) }

    // GPU temp gauge: poll framework thermals every 2 s (no sysfs access).
    var thermal by remember { mutableStateOf<ThermalMonitor.Snapshot?>(null) }
    LaunchedEffect(Unit) {
        while (true) {
            thermal = withContext(Dispatchers.IO) { ThermalMonitor.read(context) }
            kotlinx.coroutines.delay(2_000L)
        }
    }

    // Throughput: tokens/sec + cumulative tokens since server start.
    // RequestTracker is a process-wide singleton shared with the service.
    // `current` carries the in-flight request so the running total and live
    // tok/s update while generating, not just at completion.
    val reqStats by com.nickzam.server.RequestTracker.stats.collectAsState()
    val reqCurrent by com.nickzam.server.RequestTracker.current.collectAsState()
    val reqQueue by com.nickzam.server.RequestTracker.queue.collectAsState()

    // Poll active downloads + engine state. Keyed on the selected model so
    // the loop re-reads the right file/engine after a switch.
    LaunchedEffect(activeDownloads.size, status, model.id) {
        while (true) {
            val id = activeDownloads[model.filename]
            if (id != null) {
                val st = withContext(Dispatchers.IO) { downloadRepo.queryOnce(model.filename, id) }
                downloadState = st
                if (st.status == DownloadStatus.State.SUCCESSFUL ||
                    st.status == DownloadStatus.State.FAILED ||
                    st.status == DownloadStatus.State.GONE
                ) {
                    activeDownloads.remove(model.filename)
                    if (st.status == DownloadStatus.State.SUCCESSFUL) {
                        // Verify outside this keyed effect: the remove above
                        // restarts the effect and would cancel an in-place
                        // verify, wedging busy on "Verifying SHA-256…" forever.
                        busy = "Verifying SHA-256…"
                        scope.launch(Dispatchers.IO) {
                            val refusal = Sha256.verifyOrRefusal(modelFile, model.sha256)
                            withContext(Dispatchers.Main) {
                                busy = null
                                verifyNote = refusal ?: "SHA-256 verified ✓"
                            }
                        }
                    }
                }
            }
            engineLoaded = (ServiceLocator.engineRegistry?.snapshot() ?: emptyList())
                .any { it.cacheKey.startsWith("${model.id}_") }
            kotlinx.coroutines.delay(1_500L)
            if (activeDownloads.isEmpty()) {
                // Keep polling engine state lightly even with no downloads.
                kotlinx.coroutines.delay(1_500L)
            }
        }
    }

    val importLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch(Dispatchers.IO) {
            busy = "Importing…"
            try {
                context.contentResolver.openInputStream(uri)?.use { input ->
                    modelFile.outputStream().use { output -> input.copyTo(output) }
                } ?: throw IllegalStateException("Could not open selected file")
                val refusal = Sha256.verifyOrRefusal(modelFile, model.sha256)
                withContext(Dispatchers.Main) {
                    busy = null
                    verifyNote = refusal ?: "Import verified ✓ (${modelFile.length()} bytes)"
                    if (refusal != null) modelFile.delete()
                }
            } catch (e: Exception) {
                try { modelFile.delete() } catch (_: Exception) {}
                withContext(Dispatchers.Main) {
                    busy = null
                    verifyNote = "Import failed: ${e.message}"
                }
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("PixelUnlockGPU", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Text(
            "On-device Gemma · LiteRT-LM · OpenAI-compatible",
            style = MaterialTheme.typography.bodyMedium,
        )

        // ---- Server card ----
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Server", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                StatusRow(status)
                if (boundUrl != null) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            "$boundUrl/v1",
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.weight(1f),
                        )
                        OutlinedButton(onClick = { copyText(context, "$boundUrl/v1") }) {
                            Text("Copy")
                        }
                    }
                }
                if (lastError != null) {
                    Text(lastError!!, color = MaterialTheme.colorScheme.error)
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    val port = portText.toIntOrNull()?.coerceIn(1, 65535) ?: Settings.DEFAULT_PORT
                    OutlinedTextField(
                        value = portText,
                        onValueChange = { portText = it.filter { c -> c.isDigit() }.take(5) },
                        label = { Text("Port") },
                        singleLine = true,
                        modifier = Modifier.width(110.dp),
                    )
                    Button(
                        onClick = {
                            Settings.setPort(context, port)
                            ContextCompat.startForegroundService(
                                context,
                                Intent(context, NickZamServerService::class.java),
                            )
                        },
                        enabled = status != ServerState.Status.RUNNING &&
                            status != ServerState.Status.STARTING,
                    ) { Text("Start") }
                    OutlinedButton(
                        onClick = {
                            context.startService(
                                Intent(context, NickZamServerService::class.java)
                                    .setAction(NickZamServerService.ACTION_STOP),
                            )
                        },
                        enabled = status == ServerState.Status.RUNNING ||
                            status == ServerState.Status.ERROR,
                    ) { Text("Stop") }
                }
                // Exposure indicator — reflects the *running* server's bind.
                Row(verticalAlignment = Alignment.CenterVertically) {
                    val label = accessLabel
                    val raw = Settings.accessMode(context) == Settings.AccessMode.LAN &&
                        status == ServerState.Status.RUNNING
                    Text(
                        label ?: if (raw) "● RAW LAN — listening on all interfaces"
                                else "● Loopback only",
                        color = when {
                            raw -> MaterialTheme.colorScheme.error
                            label?.startsWith("TAILSCALE") == true -> MaterialTheme.colorScheme.primary
                            else -> MaterialTheme.colorScheme.primary
                        },
                        fontWeight = FontWeight.Bold,
                    )
                }
                // ---- GPU temp gauge ----
                // sysfs is off-limits to apps, so this reads the framework
                // thermal APIs (see ThermalMonitor). Real GPU °C shows only
                // if IThermalService reflection is permitted on this build.
                ThermalGaugeRow(thermal)
                ThroughputRow(reqStats, reqCurrent, reqQueue)
            }
        }

        // ---- Model card ----
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Model", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                // Picker over the catalog; selection is the default for API
                // calls without an explicit model and for /health/warm.
                Column {
                    AVAILABLE_MODELS.forEach { entry ->
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            RadioButton(
                                selected = entry.id == model.id,
                                onClick = {
                                    if (entry.id != selectedId) {
                                        selectedId = entry.id
                                        Settings.setSelectedModelId(context, entry.id)
                                        verifyNote = null
                                        downloadState = null
                                    }
                                },
                            )
                            Text(
                                "${entry.name} · ${entry.backend.name.removePrefix("LITERT_")}",
                                fontWeight = if (entry.id == model.id) FontWeight.SemiBold else FontWeight.Normal,
                            )
                        }
                    }
                }
                Text("ID: ${model.id}", style = MaterialTheme.typography.bodySmall)
                Text(
                    "Backend: ${when (model.backend) {
                        Backend.LITERT_NPU -> "Tensor G5 TPU (LITERT_NPU) · no fallback"
                        Backend.LITERT_GPU -> "OpenCL GPU delegate (LITERT_GPU) · no fallback"
                        Backend.LITERT_CPU -> "CPU (LITERT_CPU)"
                    }}",
                    style = MaterialTheme.typography.bodySmall,
                )
                SocBadgeRow(model)
                Text(
                    when {
                        installed -> "Installed: ${modelFile.length()} bytes" +
                            if (engineLoaded) " · engine LOADED" else " · engine not loaded"
                        else -> {
                            val gb = (model.expectedSizeBytes ?: 0L) / 1_000_000_000.0
                            "Not installed (~${"%.1f".format(gb)} GB download)"
                        }
                    },
                    style = MaterialTheme.typography.bodyMedium,
                )
                Text(
                    "SHA-256: ${model.sha256.take(16)}… (verified before load)",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(model.licenseNotice, style = MaterialTheme.typography.bodySmall)
                if (verifyNote != null) {
                    Text(verifyNote!!, style = MaterialTheme.typography.bodyMedium)
                }
                val dl = downloadState
                if (model.filename in activeDownloads && dl != null) {
                    LinearProgressIndicator(progress = { dl.progress }, modifier = Modifier.fillMaxWidth())
                    Text("Downloading… ${(dl.progress * 100).toInt()}%")
                }
                if (busy != null) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(modifier = Modifier.width(20.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(busy!!)
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = {
                            verifyNote = null
                            val id = downloadRepo.enqueue(model)
                            activeDownloads[model.filename] = id
                        },
                        enabled = !installed && model.filename !in activeDownloads && busy == null,
                    ) { Text("Download") }
                    OutlinedButton(
                        onClick = { importLauncher.launch(arrayOf("*/*")) },
                        enabled = busy == null && model.filename !in activeDownloads,
                    ) { Text("Import") }
                    OutlinedButton(
                        onClick = {
                            scope.launch(Dispatchers.IO) {
                                // Same lock as /v1/chat/completions: evicting while a
                                // conversation is generating frees native memory under it.
                                ServiceLocator.inferenceMutex?.withLock {
                                    ServiceLocator.sessionManager?.evictAll()
                                    ServiceLocator.engineRegistry?.evictAllLiteRt()
                                }
                                withContext(Dispatchers.Main) {
                                    engineLoaded = false
                                }
                                modelFile.delete()
                            }
                        },
                        enabled = installed && busy == null,
                    ) { Text("Delete") }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(
                        onClick = {
                            scope.launch(Dispatchers.IO) {
                                busy = "Loading engine on ${model.backend.name.removePrefix("LITERT_")}…"
                                try {
                                    // acquire can evict the LRU engine (freeing native
                                    // memory) — same lock as chat requests.
                                    ServiceLocator.inferenceMutex?.withLock {
                                        ServiceLocator.engineRegistry?.acquire(model.id, null)
                                    }
                                    withContext(Dispatchers.Main) {
                                        busy = null
                                        engineLoaded = true
                                        verifyNote = "Engine loaded on ${model.backend.name} ✓"
                                    }
                                } catch (e: Exception) {
                                    withContext(Dispatchers.Main) {
                                        busy = null
                                        verifyNote = "Load failed: ${e.message}"
                                    }
                                }
                            }
                        },
                        enabled = installed && !engineLoaded && busy == null &&
                            status == ServerState.Status.RUNNING,
                    ) { Text("Load") }
                    OutlinedButton(
                        onClick = {
                            scope.launch(Dispatchers.IO) {
                                ServiceLocator.inferenceMutex?.withLock {
                                    ServiceLocator.sessionManager?.evictAll()
                                    ServiceLocator.engineRegistry?.evictAllLiteRt()
                                }
                                withContext(Dispatchers.Main) { engineLoaded = false }
                            }
                        },
                        enabled = engineLoaded,
                    ) { Text("Unload") }
                }
            }
        }

        // ---- Generation settings card ----
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Generation", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Text(
                    "Defaults for requests that don't set these (the API can still override per-call).",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    NumberField(
                        label = "Temp",
                        value = tempText,
                        onChange = { tempText = it },
                        modifier = Modifier.weight(1f),
                    )
                    NumberField(
                        label = "Top-k",
                        value = topKText,
                        onChange = { topKText = it },
                        modifier = Modifier.width(96.dp),
                    )
                    NumberField(
                        label = "Top-p",
                        value = topPText,
                        onChange = { topPText = it },
                        modifier = Modifier.weight(1f),
                    )
                }
                // Context window: engine-baked, so applying it evicts cached
                // engines to force a rebuild at the new KV budget.
                Text(
                    "Context window: ${ctxTokens} tokens",
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    "KV-cache size per engine. Smaller = faster load + less memory, shorter chats. " +
                        "Applying rebuilds the engine.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Settings.ALLOWED_CONTEXT_TOKENS.forEach { n ->
                        val selected = n == ctxTokens
                        if (selected) {
                            Button(onClick = { }, modifier = Modifier.width(72.dp)) { Text("${n / 1024}k") }
                        } else {
                            OutlinedButton(
                                onClick = { ctxTokens = n },
                                modifier = Modifier.width(72.dp),
                            ) { Text("${n / 1024}k") }
                        }
                    }
                }
                OutlinedButton(
                    onClick = {
                        // Persist sampler defaults.
                        tempText.toFloatOrNull()?.let { Settings.setTemperature(context, it) }
                        topKText.toIntOrNull()?.let { Settings.setTopK(context, it) }
                        topPText.toFloatOrNull()?.let { Settings.setTopP(context, it) }
                        Settings.setContextTokens(context, ctxTokens)
                        // Rebuild the engine at the new context window.
                        scope.launch(Dispatchers.IO) {
                            ServiceLocator.inferenceMutex?.withLock {
                                ServiceLocator.sessionManager?.evictAll()
                                ServiceLocator.engineRegistry?.evictAllLiteRt()
                            }
                            withContext(Dispatchers.Main) {
                                engineLoaded = false
                                verifyNote = "Settings saved. Reload the model to apply the new context window."
                            }
                        }
                    },
                ) { Text("Apply") }
                Text(
                    "Note: max output length isn't a server setting — LiteRT-LM has no per-request token cap; generation runs to EOS within this window.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                // Why there is no q4/q8/full KV-precision control here, stated
                // where a user would look for it. Verified against
                // litertlm-android 0.12.0: EngineConfig/ConversationConfig
                // expose no cache-dtype field, the native engine reads
                // static k/v_cache quant scales baked into the .litertlm, and
                // Conversation is an opaque JNI handle with no cache-addressing
                // API. Quantized-KV bundles would have to be sourced and wired
                // as catalog entries, not toggled at runtime.
                Text(
                    "KV precision (q4/q8/full) isn't a runtime setting: LiteRT-LM bakes " +
                        "cache quantization into the model file, and the engine exposes no " +
                        "API to retune or address the cache per segment. This window is the " +
                        "real memory knob — it is capped at 32k, so engine budgets can never " +
                        "exceed 32k. The server reuses KV cache across chat turns, so older " +
                        "context stays cached rather than being recomputed.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        // ---- Security card ----
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Security", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                OutlinedTextField(
                    value = apiKeyText,
                    onValueChange = {
                        apiKeyText = it
                        Settings.setApiKey(context, it)
                    },
                    label = { Text("API key (empty = open)") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = {
                        val bytes = ByteArray(24).also { SecureRandom().nextBytes(it) }
                        val key = android.util.Base64.encodeToString(
                            bytes, android.util.Base64.URL_SAFE or android.util.Base64.NO_WRAP,
                        )
                        apiKeyText = key
                        Settings.setApiKey(context, key)
                    }) { Text("Generate") }
                    OutlinedButton(
                        onClick = { copyText(context, apiKeyText) },
                        enabled = apiKeyText.isNotEmpty(),
                    ) { Text("Copy") }
                    OutlinedButton(onClick = {
                        apiKeyText = ""
                        Settings.setApiKey(context, "")
                    }) { Text("Reset") }
                }
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Column(Modifier.weight(1f)) {
                        Text("Tailscale (encrypted)")
                        Text(
                            "Binds the server to this phone's tailnet IP " +
                                "(" + (Settings.tailscaleIp() ?: "not connected — install/enable Tailscale first") + "). " +
                                "Traffic is WireGuard-encrypted end-to-end; only tailnet devices can connect. " +
                                "Takes precedence over raw LAN. Restart the server after changing.",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    Switch(
                        checked = tailscaleOn,
                        onCheckedChange = {
                            tailscaleOn = it
                            Settings.setTailscale(context, it)
                        },
                    )
                }
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("Remote access — raw LAN (bind all interfaces)")
                    Switch(
                        checked = bindLan,
                        onCheckedChange = {
                            bindLan = it
                            Settings.setBindLan(context, it)
                        },
                    )
                }
                Text(
                    "Raw LAN is its own explicit opt-in and requires the bearer key above. " +
                        "It listens unencrypted on every interface — prefer the Tailscale " +
                        "toggle. While Tailscale is on it wins and raw LAN stays off. " +
                        "Restart the server after changing either.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        // ---- Client card ----
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Clients", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Text(
                    "Base URL: ${boundUrl ?: "http://127.0.0.1:${Settings.port(context)}"}/v1",
                    style = MaterialTheme.typography.bodyMedium,
                )
                if (tailscaleOn) {
                    val tip = Settings.tailscaleIp()
                    Text(
                        if (tip != null) "Tailscale URL: http://$tip:${Settings.port(context)}/v1 (from any tailnet device)"
                        else "Tailscale: on, but no tailnet IP right now — check the Tailscale app.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Text(
                    "TypingMind: Custom OpenAI endpoint + model \"${model.id}\".",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    "curl: POST /v1/chat/completions {\"model\": \"${model.id}\", " +
                        "\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun NumberField(
    label: String,
    value: String,
    onChange: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    OutlinedTextField(
        value = value,
        onValueChange = { raw ->
            // Digits, one decimal point, optional leading minus — enough to
            // type 0.8 or 40; range checks happen at Apply time on parse.
            val cleaned = raw.filter { c -> c.isDigit() || c == '.' || c == '-' }
            if (cleaned.count { it == '.' } <= 1) onChange(cleaned)
        },
        label = { Text(label) },
        singleLine = true,
        modifier = modifier,
    )
}

@Composable
private fun ThermalGaugeRow(thermal: ThermalMonitor.Snapshot?) {
    if (thermal == null) return
    val temp = thermal.gpuTempC
    val status = ThermalMonitor.statusName(thermal.status)
    // Colour escalates with thermal status; temp text is primary signal.
    val color = when (thermal.status) {
        0 -> MaterialTheme.colorScheme.primary
        1 -> MaterialTheme.colorScheme.secondary
        2 -> MaterialTheme.colorScheme.tertiary
        else -> MaterialTheme.colorScheme.error
    }
    val headroomPct = if (thermal.headroom >= 0f)
        " · headroom ${"%.0f".format(thermal.headroom * 100)}%" else ""
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = if (temp != null) "GPU temp: %.0f°C".format(temp) else "GPU temp: n/a",
            color = color,
            fontWeight = FontWeight.Bold,
        )
        Text("· thermal $status$headroomPct", color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
    if (temp == null) {
        Text(
            "Real GPU sensor blocked on this build — showing framework thermal status.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun ThroughputRow(
    stats: com.nickzam.server.RequestTracker.Stats,
    current: com.nickzam.server.RequestTracker.Entry?,
    queue: List<com.nickzam.server.RequestTracker.Entry>,
) {
    // Tick once a second while something is in flight so the live phase
    // elapsed counters move even before any token exists. The loop reads the
    // StateFlows directly so it never sees a stale key.
    var phaseNow by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(1_000)
            phaseNow = System.currentTimeMillis()
        }
    }
    fun elapsed(e: com.nickzam.server.RequestTracker.Entry?): String =
        ((e?.phaseElapsedMs(phaseNow) ?: 0L) / 1000L).toString()
    // "Since server start" = completed requests + the one in flight.
    val totalChars = stats.totalOutputChars + (current?.outputChars?.toLong() ?: 0L)
    val totalTokensEst = totalChars / 4L
    // Decode-only speed is the honest "how fast does it write" number: it
    // starts the clock at the first token, so cold-engine build + prompt
    // prefill (which can dwarf a short generation) don't drag it down. The
    // blended tokensPerSec is what a client feels end-to-end.
    val liveDecode = current?.decodeTokensPerSec ?: 0f
    val decodeTps = if (liveDecode > 0f) liveDecode else stats.avgDecodeTokensPerSec
    val running = current != null
    val phase = current?.phase
        ?: if (queue.isNotEmpty()) com.nickzam.server.RequestTracker.Phase.QUEUED else null
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                text = "%.1f tok/s".format(if (liveDecode > 0f) liveDecode else stats.avgDecodeTokensPerSec),
                color = if (phase == com.nickzam.server.RequestTracker.Phase.DECODE)
                    MaterialTheme.colorScheme.primary
                else MaterialTheme.colorScheme.onSurfaceVariant,
                fontWeight = FontWeight.Bold,
            )
            // Prefill tok/s is only honest once the whole prompt has been
            // evaluated (first token landed). Mid-window it's an upper bound
            // that decays as the denominator grows — reads like a fake
            // speedometer. While prefilling, the phase text below carries the
            // elapsed time instead.
            val prefillTps = when (phase) {
                com.nickzam.server.RequestTracker.Phase.DECODE ->
                    maxOf(current?.livePrefillTokensPerSec ?: 0f, 0f)
                        .takeIf { it > 0f } ?: stats.avgPrefillTokensPerSec
                null -> stats.avgPrefillTokensPerSec
                else -> 0f
            }
            if (prefillTps > 0f) {
                Text(
                    text = "· prefill %.0f tok/s".format(prefillTps),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Text(
                text = when (phase) {
                    com.nickzam.server.RequestTracker.Phase.QUEUED ->
                        "· queued ×${queue.size} (${elapsed(queue.first())}s)"
                    com.nickzam.server.RequestTracker.Phase.ENGINE_START ->
                        "· starting engine… ${elapsed(current)}s"
                    com.nickzam.server.RequestTracker.Phase.PREFILL ->
                        "· prefilling ≈${(current?.promptChars ?: 0) / 4} tok… ${elapsed(current)}s"
                    com.nickzam.server.RequestTracker.Phase.DECODE ->
                        "· decoding ${current?.outputTokensEst ?: 0} tok"
                    null -> if (stats.avgDecodeTokensPerSec > 0f) "· idle (decode avg)" else "· idle"
                } + if (phase != null && phase != com.nickzam.server.RequestTracker.Phase.QUEUED && queue.isNotEmpty())
                    " · ${queue.size} waiting" else "",
                color = if (phase != null && phase != com.nickzam.server.RequestTracker.Phase.QUEUED)
                    MaterialTheme.colorScheme.primary
                else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        // Post-first-token detail line: the warm-up split once it is
        // measurable. Engine build and prefill read apart from decode so a
        // slow first token reads as warm-up, not a slow model.
        val prefillMs = current?.prefillMs
        val engineMs = current?.engineStartMs
        if (prefillMs != null && prefillMs > 0) {
            Text(
                "first token: engine %.1fs + prefill %.1fs · decode %.1f tok/s".format(
                    (engineMs ?: 0L) / 1000f,
                    prefillMs / 1000f,
                    liveDecode,
                ),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Text(
            "≈%,d tokens generated since server start".format(totalTokensEst),
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            "Token counts are estimates (~4 chars/token) — LiteRT-LM reports none.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun StatusRow(status: ServerState.Status) {    val (label, color) = when (status) {
        ServerState.Status.RUNNING -> "● RUNNING" to MaterialTheme.colorScheme.primary
        ServerState.Status.STARTING -> "● STARTING…" to MaterialTheme.colorScheme.secondary
        ServerState.Status.ERROR -> "● ERROR" to MaterialTheme.colorScheme.error
        ServerState.Status.STOPPED -> "● STOPPED" to MaterialTheme.colorScheme.onSurfaceVariant
    }
    Text(label, color = color, fontWeight = FontWeight.Bold)
}

@Composable
private fun SocBadgeRow(model: ModelInfo) {
    val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        Build.SOC_MODEL ?: "unknown"
    } else "unknown (API < 31)"
    val marker = model.requiredSocMarker
    val match = socMarkerMatches(soc.takeIf { it != "unknown" }, marker)
    Text(
        "Device: ${Build.MODEL ?: "unknown"} · SoC: $soc" +
            when {
                marker == null -> " · no SoC gate for this model"
                match -> " · ✓ ${model.npuSocLabel() ?: marker} match"
                else -> " · ✗ requires ${model.npuSocLabel() ?: marker}"
            },
        style = MaterialTheme.typography.bodySmall,
        color = if (match) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
    )
}

private fun copyText(context: Context, text: String) {
    val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    cm.setPrimaryClip(ClipData.newPlainText("nickzam", text))
    Toast.makeText(context, "Copied", Toast.LENGTH_SHORT).show()
}
