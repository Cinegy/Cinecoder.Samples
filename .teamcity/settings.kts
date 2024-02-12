import jetbrains.buildServer.configs.kotlin.v2019_2.*
import jetbrains.buildServer.configs.kotlin.v2019_2.ui.*
import jetbrains.buildServer.configs.kotlin.v2019_2.buildSteps.*
import jetbrains.buildServer.configs.kotlin.v2019_2.triggers.vcs
import jetbrains.buildServer.configs.kotlin.v2019_2.vcs.GitVcsRoot
import jetbrains.buildServer.configs.kotlin.v2019_2.projectFeatures.dockerRegistry
import jetbrains.buildServer.configs.kotlin.v2019_2.buildFeatures.dockerSupport

version = "2020.1"

project {
    features {
        dockerRegistry {
            id = "CINEGY_REGISTRY"
            name = "Cinegy GitLabDocker Registry"
            url = "https://registry.cinegy.com"
            userName = "teamcity_service"
            password = "credentialsJSON:c0105573-8413-4bca-b9ca-c3cfc95d5fb2"
        }
    }

    subProject(DockerContainers)
    subProject(SampleBinaries)

}

object DockerContainers : Project({
    name = "Build Containers"
    description = "Docker container images for use with Cinecoder Samples"

    buildType(UbuntuCinecoderSamples)
})

object UbuntuCinecoderSamples : BuildType({
    name = "Ubuntu Build Container"
    description = "Docker container image for compiling Cinecoder Samples binaries on x86/x64 Linux"

    buildNumberPattern = "20.%build.counter%"

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        dockerCommand {
            name = "Build"
            commandType = build {
                source = path {
                    path = "dockerfile"
                }
                namesAndTags = """
                    registry.cinegy.com/docker/docker-builds/ubuntu2004/devcinecodersamples:latest
                    registry.cinegy.com/docker/docker-builds/ubuntu2004/devcinecodersamples:%build.number%
                """.trimIndent()
                commandArgs = "--pull"
            }
            param("dockerImage.platform", "linux")
        }
        dockerCommand {
            name = "Push"
            commandType = push {
                namesAndTags = """
                    registry.cinegy.com/docker/docker-builds/ubuntu2004/devcinecodersamples:latest
                    registry.cinegy.com/docker/docker-builds/ubuntu2004/devcinecodersamples:%build.number%
                """.trimIndent()
                removeImageAfterPush = false
            }
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "CINEGY_REGISTRY"
            }
        }
    }
})

object SampleBinaries : Project({
    name = "Sample Binaries"
    description = "Cinecoder Samples Cross-Platform Binary Builds"

    buildType(Version)
    buildType(BuildWin)
    buildType(BuildLinux)
    buildType(BuildLinuxArm64)
    buildType(BuildMacOSArm64)
    buildType(BuildAggregation)

    buildTypesOrder = arrayListOf(Version, BuildWin, BuildLinux, BuildLinuxArm64, BuildMacOSArm64, BuildAggregation)
})

object Version : BuildType({
    name = "version"
    description = "Generates a version number for use in builds"
    
    val isIntegrationBuild = DslContext.projectId.value.contains("IntegrationBuilds", ignoreCase = true)

    vcs {
        root(DslContext.settingsRoot)
        checkoutMode = CheckoutMode.ON_AGENT
        cleanCheckout = true
    }

    params {
        text("MajorVersion", "", display = ParameterDisplay.HIDDEN, allowEmpty = true) //set by version script, made as param for later steps to call easily
        text("BuildVersion", "", display = ParameterDisplay.HIDDEN, allowEmpty = true) //set by version script
        text("MinorVersion", "", display = ParameterDisplay.HIDDEN, allowEmpty = true) //set by version script
        text("SourceVersion", "", display = ParameterDisplay.HIDDEN, allowEmpty = true) //set by version script
        text("LICENSE_COMPANYNAME", "CinecoderSamples", label = "License comany name", description = "Used to set integrated Cinecoder license values", allowEmpty = false)
        //password("LICENSE_KEY", "credentialsJSON:3fdfbbdf-f8f0-43e6-a1d9-87d30c3c10d2", label = "License key", description = "Value to use for integrated Cinecoder license key", display = ParameterDisplay.HIDDEN)
    }

    steps {
        powerShell {
            name = "(patch) Generate Version Number"
            scriptMode = file {
                path = "generate_version.ps1"
            }

            // override minor version on any branches to stand out
            if (isIntegrationBuild) { //todo: correct this to actually check if this is default branch instead
                param("jetbrains_powershell_scriptArguments", "-BuildCounter %build.counter% -SourceRevisionValue %build.revisions.revision% -OverrideMinorVersion 99")
            }
            else {
                param("jetbrains_powershell_scriptArguments", "-BuildCounter %build.counter% -SourceRevisionValue %build.revisions.revision%")
            }
            
        }
    }
})

object BuildWin : BuildType({
    name = "build (win)"
    buildNumberPattern = "${Version.depParamRefs.buildNumber}"

    artifactRules = """
    _bin\Release.x64 => CinecoderSamples-Win64-%teamcity.build.branch%-%build.number%.zip
    common\cinecoder_license_string.* => LicenseIncludes-%teamcity.build.branch%-%build.number%.zip
    """.trimIndent()

	params {
        password("LICENSE_KEY", "credentialsJSON:3fdfbbdf-f8f0-43e6-a1d9-87d30c3c10d2", label = "License key", description = "Value to use for integrated Cinecoder license key", display = ParameterDisplay.HIDDEN)
    }
	
    vcs {
        root(DslContext.settingsRoot)

        checkoutMode = CheckoutMode.ON_AGENT
        cleanCheckout = true
    }

    dependencies {
        snapshot(Version) {
            reuseBuilds = ReuseBuilds.NO
        }
    }

    steps {
        powerShell {
            name = "(patch) Version (from version step)"
            scriptMode = file {
                path = "set_version.ps1"
            }
            param("jetbrains_powershell_scriptArguments", "-majorVer ${Version.depParamRefs["MajorVersion"]} -minorVer ${Version.depParamRefs["MinorVersion"]}  -buildVer ${Version.depParamRefs["BuildVersion"]}  -sourceVer ${Version.depParamRefs["SourceVersion"]}")
        }
        nuGetInstaller {
            name = "(restore) Nuget"
            toolPath = "%teamcity.tool.NuGet.CommandLine.DEFAULT%"
            projects = "Cinecoder.Samples.sln"
            param("nuget.updatePackages.mode", "perConfig")
        }
        powerShell {
            name = "(patch) Inject license"
            workingDir = "common"
            scriptMode = file {
                path = "common/inject-license.ps1"
            }
            param("jetbrains_powershell_scriptArguments", "-CompanyName ${Version.depParamRefs["LICENSE_COMPANYNAME"]} -LicenseKey %LICENSE_KEY%")
        }
         dotnetMsBuild {
            name = "(build) Samples Solution"
            projects = "Cinecoder.Samples.sln"
            version = DotnetMsBuildStep.MSBuildVersion.V17
            targets = "Build"
            configuration = "Release"
            args = "/p:Platform=x64"
        }
        // msBuild {
        //     name = "(build) Samples Solution"
        //     path = "Cinecoder.Samples.sln"
        //     version = MSBuildStep.MSBuildVersion.V14_0
        //     toolsVersion = MSBuildStep.MSBuildToolsVersion.V14_0
        //     platform = MSBuildStep.Platform.x64
        //     args = "-p:Configuration=Release"
        // }
    }
})

object BuildLinux : BuildType({
    name = "build (linux)"
    
    // check if the build type is Integration Build
    val isIntegrationBuild = DslContext.projectId.value.contains("IntegrationBuilds", ignoreCase = true)

    // Integration Builds: disable most artifacts
    if(!isIntegrationBuild)
    { 
        artifactRules = """_bin/linux => CinecoderSamples-Linux-%teamcity.build.branch%-%build.number%.zip"""
    }

	params {
        password("LICENSE_KEY", "credentialsJSON:3fdfbbdf-f8f0-43e6-a1d9-87d30c3c10d2", label = "License key", description = "Value to use for integrated Cinecoder license key", display = ParameterDisplay.HIDDEN)
    }
	
    vcs {
        root(DslContext.settingsRoot)
        checkoutMode = CheckoutMode.ON_AGENT
        cleanCheckout = true
    }

    steps {
        exec {
            name = "(patch) Version (from version step)"
            path = "pwsh"
            arguments = "./set_version.ps1 -majorVer ${Version.depParamRefs["MajorVersion"]} -minorVer ${Version.depParamRefs["MinorVersion"]}  -buildVer ${Version.depParamRefs["BuildVersion"]}  -sourceVer ${Version.depParamRefs["SourceVersion"]}"
            dockerImage = "registry.cinegy.com/docker/docker-builds/ubuntu1804/devbase:latest"
        }
        exec {
            name = "(patch) Inject license"
            path = "pwsh"
            workingDir = "common"
            arguments = "./inject-license.ps1 -CompanyName ${Version.depParamRefs["LICENSE_COMPANYNAME"]} -LicenseKey %LICENSE_KEY%"
            dockerImage = "registry.cinegy.com/docker/docker-builds/ubuntu1804/devbase:latest"
        }
        exec {
            name = "(build) Samples Script"
            path = "./build_samples-linux.sh"
            arguments = "Release"            
            dockerImage = "registry.cinegy.com/docker/docker-builds/ubuntu2004/devcinecodersamples:latest"
        }
    }

    // triggers {
    //     vcs {
    //         enabled = false
    //         branchFilter = ""
    //     }
    // }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "CINEGY_REGISTRY"
            }
        }
    }
    
    dependencies {
        snapshot(Version) {
            reuseBuilds = ReuseBuilds.NO
        }
    }
})

object BuildLinuxArm64 : BuildType({
    name = "build (linux-arm64)"
    
    // check if the build type is Integration Build
    val isIntegrationBuild = DslContext.projectId.value.contains("IntegrationBuilds", ignoreCase = true)

    // Integration Builds: disable most artifacts (adding readme so there is something in the zip to bundle)
    if(!isIntegrationBuild)
    { 
        artifactRules = """_bin/linux => CinecoderSamples-Linux-Arm64-%teamcity.build.branch%-%build.number%.zip"""
    }

	params {
        password("LICENSE_KEY", "credentialsJSON:3fdfbbdf-f8f0-43e6-a1d9-87d30c3c10d2", label = "License key", description = "Value to use for integrated Cinecoder license key", display = ParameterDisplay.HIDDEN)
    }
	
    vcs {
        root(DslContext.settingsRoot)
        checkoutMode = CheckoutMode.ON_AGENT
        cleanCheckout = true
    }

    steps {
        exec {
            name = "(patch) Version (from version step)"
            path = "pwsh"
            arguments = "./set_version.ps1 -majorVer ${Version.depParamRefs["MajorVersion"]} -minorVer ${Version.depParamRefs["MinorVersion"]}  -buildVer ${Version.depParamRefs["BuildVersion"]}  -sourceVer ${Version.depParamRefs["SourceVersion"]}"
            dockerImage = "registry.cinegy.com/docker/docker-builds/ubuntu1804/devbase:latest"
            dockerPull = true
            dockerImagePlatform = ExecBuildStep.ImagePlatform.Linux			
        }
        exec {
            name = "(patch) Inject license"
            path = "pwsh"
            workingDir = "common"
            arguments = "./inject-license.ps1 -CompanyName ${Version.depParamRefs["LICENSE_COMPANYNAME"]} -LicenseKey %LICENSE_KEY%"
            dockerImage = "registry.cinegy.com/docker/docker-builds/ubuntu1804/devbase:latest"
			dockerPull = true
            dockerImagePlatform = ExecBuildStep.ImagePlatform.Linux
        }
        exec {
            name = "(build) Samples Script"
            //enabled=false
            path = "./build_samples-linux-arm64.sh"
            arguments = "Release --platform=linux/arm64"            
            dockerImage = "registry.cinegy.com/docker/docker-builds/ubuntu2004/devbasearm64:latest"
            dockerPull = true
            dockerImagePlatform = ExecBuildStep.ImagePlatform.Linux			
        }
    }

    // triggers {
    //     vcs {
    //         enabled = false
    //         branchFilter = ""
    //     }
    // }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "CINEGY_REGISTRY"
            }
        }
    }

    dependencies {
        snapshot(Version) {
            reuseBuilds = ReuseBuilds.NO
        }
    }
})

object BuildMacOSArm64 : BuildType({
    name = "build (macos arm64)"

    // check if the build type is Integration Build
    val isIntegrationBuild = DslContext.projectId.value.contains("IntegrationBuilds", ignoreCase = true)

    // Integration Builds: disable most artifacts (adding readme so there is something in the zip to bundle)
    if(!isIntegrationBuild)
    { 
        artifactRules = """_bin/macosx => CinecoderSamples-MacOS-Arm64-%teamcity.build.branch%-%build.number%.zip"""
    }

	params {
        password("LICENSE_KEY", "credentialsJSON:3fdfbbdf-f8f0-43e6-a1d9-87d30c3c10d2", label = "License key", description = "Value to use for integrated Cinecoder license key", display = ParameterDisplay.HIDDEN)
    }

    vcs {
        root(DslContext.settingsRoot)
        checkoutMode = CheckoutMode.ON_AGENT
        cleanCheckout = true
    }

    steps {
        script {
            name = "(patch) Version (from Version Step)"
            workingDir = "."
            scriptContent = "pwsh ./set_version.ps1 -majorVer ${Version.depParamRefs["MajorVersion"]} -minorVer ${Version.depParamRefs["MinorVersion"]}  -buildVer ${Version.depParamRefs["BuildVersion"]}  -sourceVer ${Version.depParamRefs["SourceVersion"]}"
        }        
        exec {
            name = "(patch) Inject license"
            workingDir = "."
            path = "./inject_license.sh"
            arguments = "%LICENSE_COMPANYNAME% %LICENSE_KEY%"
        }
        exec {
            name = "(build) Samples Script"
            workingDir = "."
            path = "./build_samples-linux.sh"
            arguments = "Release"
        }     
    }

    // triggers {
    //     vcs {
    //         enabled = false
    //         branchFilter = ""
    //     }
    // }

    dependencies {
        snapshot(Version) {
            reuseBuilds = ReuseBuilds.NO
        }
    }

    requirements {
        moreThan("tools.xcode.version.major", "11")
        equals("teamcity.agent.jvm.os.arch", "aarch64")
    }
})

object BuildAggregation : BuildType({
    name = "build aggregation"
    description = "Collected results from all dependencies for single commit status"

    // check if the build type is Integration Build
    val isIntegrationBuild = DslContext.projectId.value.contains("IntegrationBuilds", ignoreCase = true)

    type = BuildTypeSettings.Type.COMPOSITE
    buildNumberPattern = "${Version.depParamRefs.buildNumber}"

    vcs {
        root(DslContext.settingsRoot)
        checkoutMode = CheckoutMode.ON_AGENT
        cleanCheckout = true
    }
    
    // triggers {
    //     vcs {
    //     }
    // }

    dependencies {
        dependency(BuildLinux) {
            snapshot {
            }

            artifacts {
                artifactRules = """
                    CinecoderSamples-Linux-%teamcity.build.branch%-%build.number%.zip
                """.trimIndent()
            }
        }          
        dependency(BuildLinuxArm64) {
            snapshot {
            }

            artifacts {
                artifactRules = """
                    CinecoderSamples-Linux-Arm64-%teamcity.build.branch%-%build.number%.zip
                """.trimIndent()
            }
        }        
        dependency(BuildMacOSArm64) {
            snapshot {
            }

            artifacts {
                artifactRules = """
                    CinecoderSamples-MacOS-Arm64-%teamcity.build.branch%-%build.number%.zip
                """.trimIndent()
            }
        }
        dependency(BuildWin) {
            snapshot {
            }

            artifacts {
                artifactRules = """
                    CinecoderSamples-Win64-%teamcity.build.branch%-%build.number%.zip
                    LicenseIncludes-%teamcity.build.branch%-%build.number%.zip
                """.trimIndent()
            }
        }
    }
})
