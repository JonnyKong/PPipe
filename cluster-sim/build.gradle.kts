plugins {
    id("java")
    application
}

group = "edu.purdue.dsnl"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    compileOnly("org.projectlombok:lombok:1.18.30")
    annotationProcessor("org.projectlombok:lombok:1.18.30")
    implementation("com.fasterxml.jackson.dataformat:jackson-dataformat-csv:2.14.1")
    implementation("tech.tablesaw:tablesaw-core:0.43.1")
    implementation("info.picocli:picocli:4.7.0")
    implementation("org.json:json:20230227")
    implementation("it.unimi.dsi:fastutil:8.5.12")
    implementation("org.slf4j:slf4j-nop:1.7.30")
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.1")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.8.1")
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}

application {
    mainClass.set("edu.purdue.dsnl.clustersim.Main")

    // enable assertions
    applicationDefaultJvmArgs = listOf("-ea")
}
