using BlazorApp.Components;
using MudBlazor.Services;
using Microsoft.AspNetCore.Hosting.StaticWebAssets;

var builder = WebApplication.CreateBuilder(args);
StaticWebAssetsLoader.UseStaticWebAssets(builder.Environment, builder.Configuration);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

builder.Services.AddMudServices();
builder.Services.AddHttpClient("qc", client =>
{
    var baseUrl = builder.Configuration["QcApi:BaseUrl"] ?? "http://localhost:8000";
    client.BaseAddress = new Uri(baseUrl);
    client.Timeout = TimeSpan.FromMinutes(10);
});
var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}
app.UseStatusCodePagesWithReExecute("/not-found");
app.UseHttpsRedirection();

app.UseAntiforgery();

app.UseStaticFiles();
app.MapStaticAssets();
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
