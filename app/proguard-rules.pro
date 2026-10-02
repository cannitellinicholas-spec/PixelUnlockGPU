# NickZam ProGuard / R8 rules
#
# This app uses heavy reflection through LiteRT-LM natives, Ktor, Netty and
# Gson. Without these keeps, an R8-minified release build will silently break
# at request time — long after the build succeeded.

# --- LiteRT-LM (Google AI Edge) ---------------------------------------------
-keep class com.google.ai.edge.litertlm.** { *; }
-dontwarn com.google.ai.edge.litertlm.**

# --- Ktor server + Netty ----------------------------------------------------
-keep class io.ktor.** { *; }
-dontwarn io.ktor.**
-keep class io.netty.** { *; }
-dontwarn io.netty.**
-dontwarn sun.misc.**
-dontwarn org.slf4j.**

# --- Gson -------------------------------------------------------------------
-keep class com.nickzam.server.** { *; }
-keepattributes Signature, *Annotation*, SourceFile, LineNumberTable, EnclosingMethod
-keepclassmembers,allowobfuscation class * {
    @com.google.gson.annotations.SerializedName <fields>;
}

# --- Kotlin metadata --------------------------------------------------------
-keep class kotlin.Metadata { *; }
-keepclassmembers class **$Companion { *; }
-keep class kotlin.coroutines.Continuation
-dontwarn kotlinx.coroutines.flow.**
-dontwarn kotlinx.atomicfu.**
