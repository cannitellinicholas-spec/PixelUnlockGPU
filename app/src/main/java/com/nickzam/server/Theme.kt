package com.nickzam.server

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// NickZam pastel identity: pastel blue + pastel red bolt.
private val PastelBlue = Color(0xFFA9C6FF)
private val PastelRed = Color(0xFFFF7B7B)
private val DeepBlue = Color(0xFF1E3A8A)

private val LightColors = lightColorScheme(
    primary = DeepBlue,
    secondary = PastelBlue,
    tertiary = PastelRed,
)

private val DarkColors = darkColorScheme(
    primary = PastelBlue,
    secondary = DeepBlue,
    tertiary = PastelRed,
)

@Composable
fun NickZamTheme(
    darkTheme: Boolean = androidx.compose.foundation.isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        content = content,
    )
}
