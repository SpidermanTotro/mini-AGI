package org.dragonforge.companion

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

data class HostStatus(val training: Boolean, val step: Int?, val loss: Double?,
    val experts: Int?, val vram: Double?, val doctor: String)

private const val CLIENT_API_VERSION = 1

private fun getJson(base: String, path: String): JSONObject {
    val connection = (URL(base.trimEnd('/') + path).openConnection()
        as HttpURLConnection).apply {
        connectTimeout = 4000
        readTimeout = 4000
        requestMethod = "GET"
    }
    try {
        require(connection.responseCode in 200..299) { "HTTP " + connection.responseCode }
        return JSONObject(connection.inputStream.bufferedReader().use { it.readText() })
    } finally { connection.disconnect() }
}

private suspend fun checkCompatibility(base: String) = withContext(Dispatchers.IO) {
    val json = getJson(base, "/api/v1/capabilities")
    val api = json.getInt("api_version")
    val minimum = json.optInt("api_min_client_version", api)
    require(CLIENT_API_VERSION in minimum..api) {
        "Incompatible DragonForge API: host supports " + minimum + ".." + api
    }
    require(json.optString("platform") == "linux") { "DragonForge host is not Linux" }
    val features = json.getJSONObject("features")
    require(features.optBoolean("status")) { "Host does not provide status telemetry" }
}

private suspend fun fetchStatus(base: String): HostStatus = withContext(Dispatchers.IO) {
    val json = getJson(base, "/api/v1/status")
    HostStatus(
        json.optBoolean("training", false),
        if (json.isNull("step")) null else json.getInt("step"),
        if (json.isNull("loss")) null else json.getDouble("loss"),
        if (json.isNull("experts")) null else json.getInt("experts"),
        if (json.isNull("vram_gb")) null else json.getDouble("vram_gb"),
        json.optString("doctor", "unknown"))
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { DragonForgeScreen() } }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DragonForgeScreen() {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("dragonforge", 0) }
    var host by remember { mutableStateOf(prefs.getString("linux_host", "") ?: "") }
    var status by remember { mutableStateOf<HostStatus?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()

    fun refresh() {
        if (host.isBlank()) {
            error = "Enter your Linux host address first"
            return
        }
        prefs.edit().putString("linux_host", host.trim()).apply()
        scope.launch {
            try {
                checkCompatibility(host)
                status = fetchStatus(host)
                error = null
            } catch (e: Exception) { error = e.message ?: "Connection failed" }
        }
    }

    Scaffold(topBar = { TopAppBar(title = { Text("DragonForge") }) }) { padding ->
        Column(
            Modifier.padding(padding).padding(20.dp).verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text("Samsung S21 Ultra companion", style = MaterialTheme.typography.titleMedium)
            OutlinedTextField(value = host, onValueChange = { host = it },
                label = { Text("Linux host, e.g. http://192.168.1.50:8765") }, singleLine = true,
                modifier = Modifier.fillMaxWidth())
            Button(onClick = ::refresh) { Text("Refresh") }
            status?.let {
                HorizontalDivider()
                Text("Training", style = MaterialTheme.typography.titleMedium)
                Text("State: " + if (it.training) "Running" else "Idle")
                Text("Step: " + (it.step?.toString() ?: "—"))
                Text("Loss: " + (it.loss?.toString() ?: "—"))
                Text("Experts: " + (it.experts?.toString() ?: "—"))
                Text("VRAM: " + (it.vram?.let { v -> "%.2f GB".format(v) } ?: "—"))
                HorizontalDivider()
                Text("Training Doctor", style = MaterialTheme.typography.titleMedium)
                Text("Health: " + it.doctor.replaceFirstChar(Char::uppercase))
            }
            error?.let { Text("Connection problem: " + it) }
            Text("Read-only v0.1 — training controls are disabled.")
        }
    }
}
