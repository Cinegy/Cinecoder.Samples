import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.MSBuildStep
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.exec
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.msBuild
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.nuGetInstaller
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.powerShell
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.vcs

/*
The settings script is an entry point for defining a TeamCity
project hierarchy. The script should contain a single call to the
project() function with a Project instance or an init function as
an argument.

VcsRoots, BuildTypes, Templates, and subprojects can be
registered inside the project using the vcsRoot(), buildType(),
template(), and subProject() methods respectively.

To debug settings scripts in command-line, run the

    mvnDebug org.jetbrains.teamcity:teamcity-configs-maven-plugin:generate

command and attach your debugger to the port 8000.

To debug in IntelliJ Idea, open the 'Maven Projects' tool window (View
-> Tool Windows -> Maven Projects), find the generate task node
(Plugins -> teamcity-configs -> teamcity-configs:generate), the
'Debug' option is available in the context menu for the task.
*/

version = "2019.1"

project {

    buildType(Build)
}

object Build : BuildType({
    name = "Build"

    artifactRules = """_bin\Release.x64 => CinecoderSamples-%teamcity.build.branch%-%build.number%.zip"""

    params {
        text("LICENSE_COMPANYNAME", "cinegy", label = "License comany name", description = "Used to set integrated Cinecoder license values", allowEmpty = false)
        password("LICENSE_KEY", "credentialsJSON:3232fa18-602e-4414-80d5-9901a79cfc16", label = "License key", description = "Value to use for integrated Cinecoder license key", display = ParameterDisplay.HIDDEN)
    }

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        exec {
            name = "(restore) Get libraries for build"
            path = "get-external-libraries.bat"
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
