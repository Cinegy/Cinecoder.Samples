FROM mcr.microsoft.com/dotnet/aspnet:6.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src
COPY ["SimpleBenchmark/SimpleBenchmark.csproj", "SimpleBenchmark/"]

RUN dotnet restore "SimpleBenchmark/SimpleBenchmark.csproj"
COPY . .
WORKDIR "/src/"
RUN dotnet build "SimpleBenchmark/SimpleBenchmark.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "SimpleBenchmark/SimpleBenchmark.csproj" -c Release --os linux -o /app/publish

FROM base AS final
WORKDIR /app

COPY --from=publish /app/publish .

ENTRYPOINT ["/app/SimpleBenchmark"]
