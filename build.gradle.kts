// NickZam root build. Versions are single-sourced from gradle/libs.versions.toml
// (see docs/device-model-matrix.md for the pinned compatibility set).
plugins {
    alias(libs.plugins.androidApplication) apply false
    alias(libs.plugins.kotlinAndroid) apply false
    alias(libs.plugins.kotlinCompose) apply false
}
