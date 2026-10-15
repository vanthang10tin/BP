# Multi-stage build for Blazor Server UI
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Copy project file and restore dependencies
COPY BlazorApp.csproj ./
RUN dotnet restore BlazorApp.csproj

# Copy all source files and publish
COPY . ./
RUN dotnet publish BlazorApp.csproj -c Release -o /app/publish --no-restore

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS final
WORKDIR /app
COPY --from=build /app/publish .

ENV ASPNETCORE_URLS=http://+:5000
ENV ASPNETCORE_ENVIRONMENT=Production
EXPOSE 5000

ENTRYPOINT ["dotnet", "BlazorApp.dll"]
