plugins {
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.kotlinAndroid)
    alias(libs.plugins.kotlinCompose)
}

android {
    namespace = "com.nickzam.server"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.nickzam.server"
        minSdk = 29
        targetSdk = 34
        versionCode = 3
        versionName = "0.1.2"
        // SHA-256 of the pinned model artifact (docs/device-model-matrix.md).
        // Fail-closed: a mismatch refuses the load. Updated by the Gate 0
        // audit; any artifact change must update this and re-run the gate.
        buildConfigField("String", "MODEL_SHA256", "\"af1082986639ecde7db95d91be6fe54f8b6b458104734c5bafc204e69d6852dc\"")
        buildConfigField("String", "LITERT_VERSION", "\"0.12.0\"")
    }

    signingConfigs {
        create("release") {
            // Read from ~/.gradle/gradle.properties or env. NEVER hardcode.
            val keystorePath = (findProperty("NICKZAM_KEYSTORE_PATH") as String?)
                ?: System.getenv("NICKZAM_KEYSTORE_PATH")
            val keystorePassword = (findProperty("NICKZAM_KEYSTORE_PASSWORD") as String?)
                ?: System.getenv("NICKZAM_KEYSTORE_PASSWORD")
            val keyAlias = (findProperty("NICKZAM_KEY_ALIAS") as String?)
                ?: System.getenv("NICKZAM_KEY_ALIAS")
            val keyPassword = (findProperty("NICKZAM_KEY_PASSWORD") as String?)
                ?: System.getenv("NICKZAM_KEY_PASSWORD")

            if (keystorePath != null && keystorePassword != null && keyAlias != null && keyPassword != null) {
                storeFile = file(keystorePath)
                this.storePassword = keystorePassword
                this.keyAlias = keyAlias
                this.keyPassword = keyPassword
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            val releaseCfg = signingConfigs.getByName("release")
            signingConfig = if (releaseCfg.storeFile != null) releaseCfg else signingConfigs.getByName("debug")
        }
    }

    splits {
        abi {
            isEnable = true
            reset()
            // LiteRT-LM ships JNI .so files for arm64-v8a; the Tensor G5 target
            // is arm64-only. x86/x86_64 are dropped — emulator inference is
            // unusably slow and can never prove Tensor G5 execution.
            include("arm64-v8a")
            isUniversalApk = true
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11)
        }
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    testOptions {
        unitTests {
            isIncludeAndroidResources = true
            isReturnDefaultValues = true
        }
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.0}"
            excludes += "META-INF/INDEX.LIST"
            excludes += "META-INF/io.netty.versions.properties"
        }
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material.icons.extended)
    implementation(libs.androidx.lifecycle.runtime.compose)

    // Ktor HTTP server (Netty engine) + Gson content negotiation.
    implementation(libs.ktor.server.core)
    implementation(libs.ktor.server.netty)
    implementation(libs.ktor.server.content.negotiation)
    implementation(libs.ktor.serialization.gson)

    // Inference runtime (pinned — see docs/device-model-matrix.md).
    implementation(libs.litertlm.android)

    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.robolectric)
}
