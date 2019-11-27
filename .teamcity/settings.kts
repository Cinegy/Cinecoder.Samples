import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.MSBuildStep
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.exec
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.msBuild
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.nuGetInstaller
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.powerShell
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.vcs
import jetbrains.buildServer.configs.kotlin.v2018_2.vcs.GitVcsRoot

version = "2019.1"

project {
    buildType(Version)
    buildType(Build)

    buildTypesOrder = arrayListOf(Version, Build)
}


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
        text("MajorVersion", "", display = ParameterDisplay.HIDDEN, allowEmpty = true)
        text("BuildVersion", "", display = ParameterDisplay.HIDDEN, allowEmpty = true)
        text("MinorVersion", "", display = ParameterDisplay.HIDDEN, allowEmpty = true)
        text("SourceVersion", "", display = ParameterDisplay.HIDDEN, allowEmpty = true)
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

object Build : BuildType({
    name = "Build"
    buildNumberPattern = "${Version.depParamRefs.buildNumber}"

    artifactRules = """_bin\Release.x64 => CinecoderSamples-%teamcity.build.branch%-%build.number%.zip"""

    params {
        text("LICENSE_COMPANYNAME", "cinegy", label = "License comany name", description = "Used to set integrated Cinecoder license values", allowEmpty = false)
        password("LICENSE_KEY", "zxxa438a776e2232213653a573da6eb2734dca83fc076d4ebe9c9ee46ce2dcfcf2daf64a0983187fa7e2782eea53ed0acfa9545bd6184f6c2d3ee310aba0bcd293b775d03cbe80d301b", label = "License key", description = "Value to use for integrated Cinecoder license key", display = ParameterDisplay.HIDDEN)
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
        exec {
            name = "(restore) Get libraries for build"
            path = "get-external-libraries.bat"
            enabled = false
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
            param("jetbrains_powershell_scriptArguments", "-CompanyName %LICENSE_COMPANYNAME% -LicenseKey %LICENSE_KEY%")
        }
        msBuild {
            name = "(build) Samples Solution"
            path = "Cinecoder.Samples.sln"
            version = MSBuildStep.MSBuildVersion.V14_0
            toolsVersion = MSBuildStep.MSBuildToolsVersion.V14_0
            platform = MSBuildStep.Platform.x64
            args = "-p:Configuration=Release"
        }
    }

    triggers {
        vcs {
        }
    }
})