using System.Reflection;
using System.Text.Json;
using BlazorApp.Components.Pages;
using Microsoft.AspNetCore.Components.Forms;

var type = typeof(Home);
object? Call(string name, object? instance, params object[] args) => type.GetMethod(name,
    BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static)!.Invoke(instance, args);
object? Field(Home home, string name) => type.GetField(name, BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(home);
void Check(bool condition, string message) { if (!condition) throw new Exception(message); }
JsonElement Json(string text) => JsonSerializer.Deserialize<JsonElement>(text);

foreach (var input in new[] { "null", "true", "[]", "{}", "\"unknown\"" })
    Check((int)Call("GetInt", null, Json($"{{\"count\":{input}}}"), "count", -1)! == -1, "Invalid numeric JSON must not kill rendering");
Check((int)Call("GetInt", null, Json("{\"count\":\"42\"}"), "count", -1)! == 42, "String counts supported");
Check(Call("GetPredictionBox", null, Json("{\"box\":[null,0,20,20]}")) is null, "Null coordinates skipped");
Check(Call("GetBox", null, Json("null")) is null, "Null annotations skipped");
Check(!(bool)Call("ObjectIdMatches", null, "10", "1")!, "Object 1 must not match object 10");
Check((bool)Call("ObjectIdMatches", null, "1 & 2", "1")!, "Duplicate-object evidence still matches");

var polygonBox = ((double X, double Y, double W, double H)?)Call("GetPredictionBox", null,
    Json("{\"type\":\"polygon\",\"points\":[10,10,30,10,30,40,10,40]}"));
Check(polygonBox == (10d, 10d, 20d, 30d), "Polygon bounds must include every vertex");

var home = new Home();
async Task Upload(params IBrowserFile[] files) => await (Task)Call("OnImagesSelected", home, new InputFileChangeEventArgs(files))!;
await Upload(Enumerable.Range(0, 201).Select(i => (IBrowserFile)new FakeFile($"{i}.png", 1)).ToArray());
Check((bool)Field(home, "HasError")! && !(bool)Field(home, "IsBusy")!, "Too many uploads must show an error and unlock controls");
await Upload(new FakeFile("large.png", 11 * 1024 * 1024));
Check((bool)Field(home, "HasError")! && !(bool)Field(home, "IsBusy")!, "Oversized uploads must unlock controls");
await Upload(new FakeFile("first.png", 1), new FakeFile("broken.png", 1, true));
Check(((Dictionary<string, string>)Field(home, "ImageFiles")!).Count == 0, "Failed uploads must not leave partial media");
await Upload(new FakeFile("valid.png", 1));
Check(!(bool)Field(home, "HasError")! && ((Dictionary<string, string>)Field(home, "ImageFiles")!).Count == 1, "Valid upload must work after previous errors");
Console.WriteLine("Blazor regression checks passed: numeric JSON, geometry, object IDs, upload limits and recovery.");

sealed class FakeFile(string name, long size, bool fail = false) : IBrowserFile
{
    public string Name => name;
    public DateTimeOffset LastModified => DateTimeOffset.UtcNow;
    public long Size => size;
    public string ContentType => "image/png";
    public Stream OpenReadStream(long maxAllowedSize = 512000, CancellationToken cancellationToken = default)
        => fail ? throw new IOException("simulated interrupted upload") : new MemoryStream(new byte[] { 1 });
}
