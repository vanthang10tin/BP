using BlazorApp.Components;
using MudBlazor.Services;
using Microsoft.AspNetCore.Hosting.StaticWebAssets;
using Microsoft.AspNetCore.DataProtection;
using Microsoft.AspNetCore.HttpOverrides;

var builder = WebApplication.CreateBuilder(args);
var customPort = Environment.GetEnvironmentVariable("PORT");
builder.WebHost.UseUrls($"http://*:{customPort ?? "5000"}");

if (builder.Environment.IsDevelopment())
{
    StaticWebAssetsLoader.UseStaticWebAssets(builder.Environment, builder.Configuration);
}

// Add services to the container.
builder.Services.AddDataProtection()
    .SetApplicationName("QCStudio2D");

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents(options =>
    {
        options.DetailedErrors = builder.Environment.IsDevelopment();
        options.DisconnectedCircuitRetentionPeriod = TimeSpan.FromMinutes(5);
        options.DisconnectedCircuitMaxRetained = 100;
    }).AddHubOptions(options =>
{
    // Generous limit for SignalR message chunks
    options.MaximumReceiveMessageSize = 10 * 1024 * 1024;
});

builder.Services.AddMudServices();
builder.Services.AddHttpClient("qc", client =>
{
    var baseUrl = builder.Configuration["QcApi:BaseUrl"] ?? "http://127.0.0.1:8000";
    client.BaseAddress = new Uri(baseUrl);
    client.Timeout = TimeSpan.FromMinutes(30);
});
var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
}

// Forwarded headers for reverse proxy (Render, Cloudflare, Nginx)
var forwardedOptions = new ForwardedHeadersOptions
{
    ForwardedHeaders = ForwardedHeaders.XForwardedFor | ForwardedHeaders.XForwardedProto
};
forwardedOptions.KnownIPNetworks.Clear();
forwardedOptions.KnownProxies.Clear();
app.UseForwardedHeaders(forwardedOptions);

app.UseStatusCodePagesWithReExecute("/not-found");

app.UseStaticFiles();
app.MapStaticAssets();

app.UseAntiforgery();

app.MapGet("/healthz", async (IHttpClientFactory clients, CancellationToken cancellationToken) =>
{
    using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
    timeout.CancelAfter(TimeSpan.FromSeconds(5));
    try
    {
        using var response = await clients.CreateClient("qc").GetAsync("/api/health", timeout.Token);
        return response.IsSuccessStatusCode
            ? Results.Ok(new { status = "ok" })
            : Results.Json(new { status = "degraded", code = (int)response.StatusCode }, statusCode: StatusCodes.Status503ServiceUnavailable);
    }
    catch (Exception ex)
    {
        return Results.Json(new { status = "unavailable", error = ex.Message }, statusCode: StatusCodes.Status503ServiceUnavailable);
    }
});

// Keep frame previews same-origin with the Blazor UI. The browser must not
// assume that the QC API is listening on its own localhost when deployed.
app.MapGet("/qc-image/{filename}", async (string filename, IHttpClientFactory clients, CancellationToken cancellationToken) =>
{
    if (string.IsNullOrWhiteSpace(filename) || filename != Path.GetFileName(filename) ||
        !filename.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) &&
        !filename.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) &&
        !filename.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
        return Results.BadRequest();

    try
    {
        var client = clients.CreateClient("qc");
        using var response = await client.GetAsync($"/api/image/{Uri.EscapeDataString(filename)}", cancellationToken);
        if (!response.IsSuccessStatusCode)
            return Results.StatusCode((int)response.StatusCode);
        var bytes = await response.Content.ReadAsByteArrayAsync(cancellationToken);
        var contentType = response.Content.Headers.ContentType?.MediaType ?? "application/octet-stream";
        return Results.File(bytes, contentType);
    }
    catch (HttpRequestException)
    {
        return Results.StatusCode(StatusCodes.Status503ServiceUnavailable);
    }
});

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
