FROM ubuntu:18.04

LABEL maintainer="Lewis Kirkaldie <lewis@cinegy.com>"

ENV TZ=Europe/Berlin
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#base dev packages and OS tools, including mono and boost other library dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
        build-essential git nano gnupg apt-utils unzip sudo bzip2 \
        apt-transport-https curl wget ca-certificates cpio \
        mono-complete libboost-dev uuid-dev libssl-dev software-properties-common \
        jq libopenal-dev freeglut3-dev libglew-dev

#add powershell core & dotnet 3
RUN cd /tmp && \
  curl -o packages-microsoft-prod.deb https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb -sS && \
  dpkg -i packages-microsoft-prod.deb && \
  rm packages-microsoft-prod.deb && \
  apt-get update && \
  apt-get install -y powershell && \
  apt-get install -y dotnet-sdk-3.0 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Download v4.9.4 `nuget.exe` to `/usr/local/bin` and alias (later nuget seems to not like the 18.04 ubuntu mono version)
RUN curl -o /usr/local/bin/nuget.exe https://dist.nuget.org/win-x86-commandline/v4.9.4/nuget.exe && \
alias nuget="mono /usr/local/bin/nuget.exe"

#add cmake 3.15.5
RUN mkdir -p /usr/src/cmake \
&& curl -L https://github.com/Kitware/CMake/releases/download/v3.15.5/cmake-3.15.5-Linux-x86_64.tar.gz -o /usr/src/cmake/cmake.tar.gz -sS \
&& tar -C /usr/src/cmake -xvzf /usr/src/cmake/cmake.tar.gz \
&& ln -s /usr/src/cmake/cmake-3.15.5-Linux-x86_64/bin/cmake /usr/bin/cmake \
&& rm /usr/src/cmake/cmake.tar.gz

#add tini, for managing container init
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
SHELL ["/bin/bash", "-c"]