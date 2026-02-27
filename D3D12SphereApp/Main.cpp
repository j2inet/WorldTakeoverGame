// Main.cpp - Direct3D 12 application rendering a rotating sphere with checkerboard
// texture in front of a square with solid color #8080FF.
// Press ESC to exit. Resolution: 1920x1080.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include "include/directx/d3dx12.h"
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <wrl/client.h>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;
using namespace DirectX;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const UINT WIDTH  = 1920;
static const UINT HEIGHT = 1080;
static const UINT FRAME_COUNT = 2;

// ---------------------------------------------------------------------------
// Vertex layout
// ---------------------------------------------------------------------------
struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT2 texcoord;
};

// ---------------------------------------------------------------------------
// Per-object constant buffer (one MVP matrix)
// ---------------------------------------------------------------------------
struct ConstantBufferData
{
    XMFLOAT4X4 mvp;
    float       pad[48]; // pad to 256 bytes
};
static_assert(sizeof(ConstantBufferData) == 256, "CB must be 256 bytes");

// ---------------------------------------------------------------------------
// Helper: throw on HRESULT failure
// ---------------------------------------------------------------------------
static void ThrowIfFailed(HRESULT hr)
{
    if (FAILED(hr))
        throw std::runtime_error("HRESULT failed");
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
static void BuildQuad(std::vector<Vertex>& verts, std::vector<UINT>& indices)
{
    // A square in the XY plane at z = 0, covering [-1.5, 1.5]
    const float s = 1.5f;
    verts = {
        { {-s, -s, 0.0f}, {0.0f, 1.0f} },
        { {-s,  s, 0.0f}, {0.0f, 0.0f} },
        { { s,  s, 0.0f}, {1.0f, 0.0f} },
        { { s, -s, 0.0f}, {1.0f, 1.0f} },
    };
    indices = { 0, 1, 2,  0, 2, 3 };
}

static void BuildSphere(std::vector<Vertex>& verts, std::vector<UINT>& indices,
                        float radius, UINT stacks, UINT slices)
{
    verts.clear();
    indices.clear();

    for (UINT i = 0; i <= stacks; ++i)
    {
        float phi = XM_PI * float(i) / float(stacks); // 0 .. PI
        for (UINT j = 0; j <= slices; ++j)
        {
            float theta = XM_2PI * float(j) / float(slices); // 0 .. 2PI
            Vertex v;
            v.position = {
                radius * sinf(phi) * cosf(theta),
                radius * cosf(phi),
                radius * sinf(phi) * sinf(theta)
            };
            v.texcoord = { float(j) / float(slices), float(i) / float(stacks) };
            verts.push_back(v);
        }
    }

    for (UINT i = 0; i < stacks; ++i)
    {
        for (UINT j = 0; j < slices; ++j)
        {
            UINT row0 = i * (slices + 1);
            UINT row1 = (i + 1) * (slices + 1);
            indices.push_back(row0 + j);
            indices.push_back(row1 + j);
            indices.push_back(row0 + j + 1);

            indices.push_back(row1 + j);
            indices.push_back(row1 + j + 1);
            indices.push_back(row0 + j + 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Texture generators
// ---------------------------------------------------------------------------
// Returns RGBA8 data (WIDTH x HEIGHT)
static std::vector<UINT> MakeSolidTexture(UINT w, UINT h, UINT rgba)
{
    return std::vector<UINT>(w * h, rgba);
}

static std::vector<UINT> MakeCheckerboardTexture(UINT w, UINT h, UINT cells)
{
    std::vector<UINT> data(w * h);
    for (UINT y = 0; y < h; ++y)
    {
        for (UINT x = 0; x < w; ++x)
        {
            UINT cx = (x * cells) / w;
            UINT cy = (y * cells) / h;
            bool white = (cx + cy) % 2 == 0;
            data[y * w + x] = white ? 0xFFFFFFFF : 0xFF000000;
        }
    }
    return data;
}

// ---------------------------------------------------------------------------
// Main application class
// ---------------------------------------------------------------------------
class D3D12SphereApp
{
public:
    D3D12SphereApp() = default;
    ~D3D12SphereApp() { WaitForGpu(); }

    void Init(HWND hwnd);
    void Render();
    void OnKeyDown(WPARAM key);

    bool ShouldQuit() const { return m_quit; }

private:
    // D3D12 core
    ComPtr<ID3D12Device>               m_device;
    ComPtr<IDXGISwapChain3>            m_swapChain;
    ComPtr<ID3D12CommandQueue>         m_cmdQueue;
    ComPtr<ID3D12CommandAllocator>     m_cmdAlloc[FRAME_COUNT];
    ComPtr<ID3D12GraphicsCommandList>  m_cmdList;
    ComPtr<ID3D12DescriptorHeap>       m_rtvHeap;
    ComPtr<ID3D12DescriptorHeap>       m_dsvHeap;
    ComPtr<ID3D12DescriptorHeap>       m_srvHeap;
    ComPtr<ID3D12Resource>             m_renderTargets[FRAME_COUNT];
    ComPtr<ID3D12Resource>             m_depthStencil;

    // Pipeline
    ComPtr<ID3D12RootSignature>        m_rootSig;
    ComPtr<ID3D12PipelineState>        m_pso;

    // Sync
    ComPtr<ID3D12Fence>                m_fence;
    UINT64                             m_fenceValue[FRAME_COUNT] = {};
    UINT64                             m_currentFence = 0;
    HANDLE                             m_fenceEvent = nullptr;
    UINT                               m_frameIndex = 0;

    // Descriptor sizes
    UINT m_rtvDescSize = 0;

    // Meshes
    ComPtr<ID3D12Resource> m_quadVB, m_quadIB;
    D3D12_VERTEX_BUFFER_VIEW m_quadVBV{};
    D3D12_INDEX_BUFFER_VIEW  m_quadIBV{};
    UINT m_quadIndexCount = 0;

    ComPtr<ID3D12Resource> m_sphereVB, m_sphereIB;
    D3D12_VERTEX_BUFFER_VIEW m_sphereVBV{};
    D3D12_INDEX_BUFFER_VIEW  m_sphereIBV{};
    UINT m_sphereIndexCount = 0;

    // Textures (solid + checkerboard)
    ComPtr<ID3D12Resource> m_solidTex;
    ComPtr<ID3D12Resource> m_checkerTex;

    // Constant buffers (2 objects x FRAME_COUNT frames)
    ComPtr<ID3D12Resource> m_cbResource;
    ConstantBufferData*    m_cbMapped = nullptr;

    // Camera / projection
    XMMATRIX m_proj;
    XMMATRIX m_view;

    float m_rotation = 0.0f;
    bool  m_quit = false;

    // Helpers
    void LoadPipeline(HWND hwnd);
    void LoadAssets();
    void WaitForGpu();
    void MoveToNextFrame();

    ComPtr<ID3D12Resource> CreateTextureFromData(
        const UINT* pixels, UINT w, UINT h,
        ComPtr<ID3D12Resource>& uploadBuf,
        ID3D12GraphicsCommandList* cmdList);

    ComPtr<ID3D12Resource> CreateBufferUpload(UINT64 size, const void* initData);

    void RecordCommands();
};

// ---------------------------------------------------------------------------
// Win32 window procedure
// ---------------------------------------------------------------------------
static D3D12SphereApp* g_app = nullptr;

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg)
    {
    case WM_KEYDOWN:
        if (g_app) g_app->OnKeyDown(wp);
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    }
    return DefWindowProc(hwnd, msg, wp, lp);
}

// ---------------------------------------------------------------------------
// D3D12SphereApp::Init
// ---------------------------------------------------------------------------
void D3D12SphereApp::Init(HWND hwnd)
{
    LoadPipeline(hwnd);
    LoadAssets();
}

// ---------------------------------------------------------------------------
void D3D12SphereApp::LoadPipeline(HWND hwnd)
{
    // Debug layer (optional, harmless in release)
#if defined(_DEBUG)
    {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
            debug->EnableDebugLayer();
    }
#endif

    ComPtr<IDXGIFactory4> factory;
    ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));

    // Device
    ThrowIfFailed(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                    IID_PPV_ARGS(&m_device)));

    // Command queue
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ThrowIfFailed(m_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_cmdQueue)));

    // Swap chain
    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.BufferCount  = FRAME_COUNT;
    scd.Width        = WIDTH;
    scd.Height       = HEIGHT;
    scd.Format       = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage  = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.SwapEffect   = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.SampleDesc.Count = 1;

    ComPtr<IDXGISwapChain1> sc1;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(m_cmdQueue.Get(), hwnd,
                                                  &scd, nullptr, nullptr, &sc1));
    ThrowIfFailed(sc1.As(&m_swapChain));
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

    // Descriptor heaps
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.NumDescriptors = FRAME_COUNT;
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        ThrowIfFailed(m_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_rtvHeap)));
        m_rtvDescSize = m_device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    }
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.NumDescriptors = 1;
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        ThrowIfFailed(m_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_dsvHeap)));
    }
    {
        // 2 SRVs (solid + checker textures)
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.NumDescriptors = 2;
        hd.Type  = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(m_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_srvHeap)));
    }

    // RTVs
    {
        CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
            m_rtvHeap->GetCPUDescriptorHandleForHeapStart());
        for (UINT i = 0; i < FRAME_COUNT; ++i)
        {
            ThrowIfFailed(m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_renderTargets[i])));
            m_device->CreateRenderTargetView(m_renderTargets[i].Get(), nullptr, rtvHandle);
            rtvHandle.Offset(1, m_rtvDescSize);
        }
    }

    // Depth stencil
    {
        D3D12_HEAP_PROPERTIES hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        D3D12_RESOURCE_DESC   rd = CD3DX12_RESOURCE_DESC::Tex2D(
            DXGI_FORMAT_D32_FLOAT, WIDTH, HEIGHT, 1, 0, 1, 0,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        D3D12_CLEAR_VALUE cv{};
        cv.Format = DXGI_FORMAT_D32_FLOAT;
        cv.DepthStencil.Depth = 1.0f;
        ThrowIfFailed(m_device->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv,
            IID_PPV_ARGS(&m_depthStencil)));
        m_device->CreateDepthStencilView(
            m_depthStencil.Get(), nullptr,
            m_dsvHeap->GetCPUDescriptorHandleForHeapStart());
    }

    // Command allocators & list
    for (UINT i = 0; i < FRAME_COUNT; ++i)
        ThrowIfFailed(m_device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_cmdAlloc[i])));

    ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        m_cmdAlloc[m_frameIndex].Get(), nullptr, IID_PPV_ARGS(&m_cmdList)));
    ThrowIfFailed(m_cmdList->Close());

    // Fence
    ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                        IID_PPV_ARGS(&m_fence)));
    m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    for (UINT i = 0; i < FRAME_COUNT; ++i) m_fenceValue[i] = 0;
}

// ---------------------------------------------------------------------------
// Helper: upload buffer
// ---------------------------------------------------------------------------
ComPtr<ID3D12Resource> D3D12SphereApp::CreateBufferUpload(UINT64 size, const void* initData)
{
    ComPtr<ID3D12Resource> buf;
    D3D12_HEAP_PROPERTIES hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC   rd = CD3DX12_RESOURCE_DESC::Buffer(size);
    ThrowIfFailed(m_device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&buf)));

    if (initData)
    {
        void* mapped = nullptr;
        D3D12_RANGE readRange{0, 0};
        ThrowIfFailed(buf->Map(0, &readRange, &mapped));
        memcpy(mapped, initData, size);
        buf->Unmap(0, nullptr);
    }
    return buf;
}

// ---------------------------------------------------------------------------
// Helper: create texture from RGBA8 pixel data
// ---------------------------------------------------------------------------
ComPtr<ID3D12Resource> D3D12SphereApp::CreateTextureFromData(
    const UINT* pixels, UINT w, UINT h,
    ComPtr<ID3D12Resource>& uploadBuf,
    ID3D12GraphicsCommandList* cmdList)
{
    // Default heap texture
    ComPtr<ID3D12Resource> tex;
    D3D12_HEAP_PROPERTIES hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_RESOURCE_DESC   rd = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R8G8B8A8_UNORM, w, h, 1, 1);
    ThrowIfFailed(m_device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tex)));

    // Upload heap buffer
    UINT64 uploadSize = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    UINT numRows = 0;
    UINT64 rowSizeBytes = 0;
    m_device->GetCopyableFootprints(&rd, 0, 1, 0,
        &layout, &numRows, &rowSizeBytes, &uploadSize);

    D3D12_HEAP_PROPERTIES upHp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC   upRd = CD3DX12_RESOURCE_DESC::Buffer(uploadSize);
    ThrowIfFailed(m_device->CreateCommittedResource(
        &upHp, D3D12_HEAP_FLAG_NONE, &upRd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuf)));

    // Copy pixel rows into upload buffer respecting row pitch
    BYTE* dest = nullptr;
    D3D12_RANGE readRange{0, 0};
    ThrowIfFailed(uploadBuf->Map(0, &readRange, reinterpret_cast<void**>(&dest)));
    dest += layout.Offset;
    for (UINT row = 0; row < h; ++row)
    {
        memcpy(dest + row * layout.Footprint.RowPitch,
               pixels + row * w,
               w * sizeof(UINT));
    }
    uploadBuf->Unmap(0, nullptr);

    // Copy upload -> default texture
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource        = tex.Get();
    dst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource       = uploadBuf.Get();
    src.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = layout;

    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    // Transition to shader resource
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        tex.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    cmdList->ResourceBarrier(1, &barrier);

    return tex;
}

// ---------------------------------------------------------------------------
void D3D12SphereApp::LoadAssets()
{
    // ----- Root signature -----
    {
        D3D12_DESCRIPTOR_RANGE1 srvRange{};
        srvRange.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srvRange.NumDescriptors                    = 1;
        srvRange.BaseShaderRegister                = 0;
        srvRange.RegisterSpace                     = 0;
        srvRange.OffsetInDescriptorsFromTableStart = 0;
        srvRange.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;

        D3D12_ROOT_PARAMETER1 params[2]{};
        // param 0: CBV (b0) - inline root CBV
        params[0].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[0].Descriptor.ShaderRegister = 0;
        params[0].Descriptor.RegisterSpace  = 0;
        params[0].Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC;
        params[0].ShaderVisibility          = D3D12_SHADER_VISIBILITY_VERTEX;

        // param 1: SRV descriptor table (t0)
        params[1].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable.NumDescriptorRanges = 1;
        params[1].DescriptorTable.pDescriptorRanges   = &srvRange;
        params[1].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_STATIC_SAMPLER_DESC sampler{};
        sampler.Filter           = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.AddressV         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.AddressW         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.ShaderRegister   = 0;
        sampler.RegisterSpace    = 0;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC rsd{};
        rsd.Version                    = D3D_ROOT_SIGNATURE_VERSION_1_1;
        rsd.Desc_1_1.NumParameters     = 2;
        rsd.Desc_1_1.pParameters       = params;
        rsd.Desc_1_1.NumStaticSamplers = 1;
        rsd.Desc_1_1.pStaticSamplers   = &sampler;
        rsd.Desc_1_1.Flags             =
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        ComPtr<ID3DBlob> sigBlob, errBlob;
        ThrowIfFailed(D3D12SerializeVersionedRootSignature(&rsd,
                                                           &sigBlob, &errBlob));
        ThrowIfFailed(m_device->CreateRootSignature(
            0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(),
            IID_PPV_ARGS(&m_rootSig)));
    }

    // ----- Compile shaders -----
    ComPtr<ID3DBlob> vsBlob, psBlob, errBlob;
    UINT compileFlags = 0;
#if defined(_DEBUG)
    compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    // Get shader file path relative to executable
    WCHAR exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    // Replace exe name with shader filename
    WCHAR* lastSlash = wcsrchr(exePath, L'\\');
    if (lastSlash) *(lastSlash + 1) = L'\0';
    std::wstring shaderPath = std::wstring(exePath) + L"shaders.hlsl";

    HRESULT hrVS = D3DCompileFromFile(shaderPath.c_str(), nullptr, nullptr,
        "VSMain", "vs_5_0", compileFlags, 0, &vsBlob, &errBlob);
    if (FAILED(hrVS))
    {
        if (errBlob)
            OutputDebugStringA((char*)errBlob->GetBufferPointer());
        ThrowIfFailed(hrVS);
    }

    HRESULT hrPS = D3DCompileFromFile(shaderPath.c_str(), nullptr, nullptr,
        "PSMain", "ps_5_0", compileFlags, 0, &psBlob, &errBlob);
    if (FAILED(hrPS))
    {
        if (errBlob)
            OutputDebugStringA((char*)errBlob->GetBufferPointer());
        ThrowIfFailed(hrPS);
    }

    // ----- PSO -----
    D3D12_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0,
          D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 12,
          D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.InputLayout     = { layout, _countof(layout) };
    psoDesc.pRootSignature  = m_rootSig.Get();
    psoDesc.VS              = CD3DX12_SHADER_BYTECODE(vsBlob.Get());
    psoDesc.PS              = CD3DX12_SHADER_BYTECODE(psBlob.Get());
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    psoDesc.BlendState      = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    psoDesc.SampleMask      = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0]   = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.DSVFormat       = DXGI_FORMAT_D32_FLOAT;
    psoDesc.SampleDesc.Count = 1;
    ThrowIfFailed(m_device->CreateGraphicsPipelineState(&psoDesc,
                                                        IID_PPV_ARGS(&m_pso)));

    // ----- Upload resources using a temporary command list -----
    ThrowIfFailed(m_cmdAlloc[m_frameIndex]->Reset());
    ThrowIfFailed(m_cmdList->Reset(m_cmdAlloc[m_frameIndex].Get(), m_pso.Get()));

    // Keep upload buffers alive until GPU is done
    ComPtr<ID3D12Resource> solidUpload, checkerUpload;

    // Textures
    {
        auto solidPixels = MakeSolidTexture(256, 256, 0xFFFF8080); // RGBA: R=0x80 G=0x80 B=0xFF -> ABGR in mem
        // Note: R8G8B8A8_UNORM stores bytes as R,G,B,A.
        // #8080FF in HTML = R=0x80, G=0x80, B=0xFF
        // As UINT (little-endian): ABGR = 0xFF FF 80 80 -> 0xFFFF8080
        m_solidTex = CreateTextureFromData(solidPixels.data(), 256, 256,
                                           solidUpload, m_cmdList.Get());

        auto checkerPixels = MakeCheckerboardTexture(256, 256, 16);
        m_checkerTex = CreateTextureFromData(checkerPixels.data(), 256, 256,
                                             checkerUpload, m_cmdList.Get());
    }

    // Create SRVs
    {
        UINT srvSize = m_device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        CD3DX12_CPU_DESCRIPTOR_HANDLE h(
            m_srvHeap->GetCPUDescriptorHandleForHeapStart());

        D3D12_SHADER_RESOURCE_VIEW_DESC srvd{};
        srvd.Format                    = DXGI_FORMAT_R8G8B8A8_UNORM;
        srvd.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvd.Shader4ComponentMapping   = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvd.Texture2D.MipLevels       = 1;

        // SRV slot 0 = solid texture (quad / square)
        m_device->CreateShaderResourceView(m_solidTex.Get(), &srvd, h);
        h.Offset(1, srvSize);

        // SRV slot 1 = checkerboard texture (sphere)
        m_device->CreateShaderResourceView(m_checkerTex.Get(), &srvd, h);
    }

    // Quad mesh
    {
        std::vector<Vertex> verts; std::vector<UINT> indices;
        BuildQuad(verts, indices);
        m_quadIndexCount = (UINT)indices.size();

        m_quadVB = CreateBufferUpload(verts.size() * sizeof(Vertex), verts.data());
        m_quadIB = CreateBufferUpload(indices.size() * sizeof(UINT), indices.data());

        m_quadVBV.BufferLocation = m_quadVB->GetGPUVirtualAddress();
        m_quadVBV.SizeInBytes    = (UINT)(verts.size() * sizeof(Vertex));
        m_quadVBV.StrideInBytes  = sizeof(Vertex);

        m_quadIBV.BufferLocation = m_quadIB->GetGPUVirtualAddress();
        m_quadIBV.SizeInBytes    = (UINT)(indices.size() * sizeof(UINT));
        m_quadIBV.Format         = DXGI_FORMAT_R32_UINT;
    }

    // Sphere mesh
    {
        std::vector<Vertex> verts; std::vector<UINT> indices;
        BuildSphere(verts, indices, 0.8f, 40, 40);
        m_sphereIndexCount = (UINT)indices.size();

        m_sphereVB = CreateBufferUpload(verts.size() * sizeof(Vertex), verts.data());
        m_sphereIB = CreateBufferUpload(indices.size() * sizeof(UINT), indices.data());

        m_sphereVBV.BufferLocation = m_sphereVB->GetGPUVirtualAddress();
        m_sphereVBV.SizeInBytes    = (UINT)(verts.size() * sizeof(Vertex));
        m_sphereVBV.StrideInBytes  = sizeof(Vertex);

        m_sphereIBV.BufferLocation = m_sphereIB->GetGPUVirtualAddress();
        m_sphereIBV.SizeInBytes    = (UINT)(indices.size() * sizeof(UINT));
        m_sphereIBV.Format         = DXGI_FORMAT_R32_UINT;
    }

    // Constant buffer (2 objects * FRAME_COUNT, 256-byte aligned)
    {
        UINT cbSize = sizeof(ConstantBufferData) * 2 * FRAME_COUNT;
        m_cbResource = CreateBufferUpload(cbSize, nullptr);

        D3D12_RANGE readRange{0, 0};
        ThrowIfFailed(m_cbResource->Map(0, &readRange,
                                        reinterpret_cast<void**>(&m_cbMapped)));
    }

    // Camera
    m_view = XMMatrixLookAtLH(
        XMVectorSet(0, 0, -5.0f, 1),
        XMVectorSet(0, 0,  0,    1),
        XMVectorSet(0, 1,  0,    0));
    m_proj = XMMatrixPerspectiveFovLH(
        XMConvertToRadians(45.0f),
        float(WIDTH) / float(HEIGHT),
        0.1f, 100.0f);

    // Execute init commands
    ThrowIfFailed(m_cmdList->Close());
    ID3D12CommandList* lists[] = { m_cmdList.Get() };
    m_cmdQueue->ExecuteCommandLists(1, lists);
    WaitForGpu();
}

// ---------------------------------------------------------------------------
void D3D12SphereApp::OnKeyDown(WPARAM key)
{
    if (key == VK_ESCAPE)
        m_quit = true;
}

// ---------------------------------------------------------------------------
void D3D12SphereApp::WaitForGpu()
{
    ThrowIfFailed(m_cmdQueue->Signal(m_fence.Get(), ++m_currentFence));
    ThrowIfFailed(m_fence->SetEventOnCompletion(m_currentFence, m_fenceEvent));
    WaitForSingleObject(m_fenceEvent, INFINITE);
    m_fenceValue[m_frameIndex] = m_currentFence;
}

// ---------------------------------------------------------------------------
void D3D12SphereApp::MoveToNextFrame()
{
    // Signal end of current frame
    const UINT64 currentFenceVal = ++m_currentFence;
    ThrowIfFailed(m_cmdQueue->Signal(m_fence.Get(), currentFenceVal));
    m_fenceValue[m_frameIndex] = currentFenceVal;

    // Update frame index
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

    // Wait if next frame's resources are still in use
    if (m_fence->GetCompletedValue() < m_fenceValue[m_frameIndex])
    {
        ThrowIfFailed(m_fence->SetEventOnCompletion(
            m_fenceValue[m_frameIndex], m_fenceEvent));
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
}

// ---------------------------------------------------------------------------
void D3D12SphereApp::RecordCommands()
{
    // Update rotation
    m_rotation += 0.01f;
    if (m_rotation > XM_2PI) m_rotation -= XM_2PI;

    // CB base for this frame
    UINT cbOffset = m_frameIndex * 2;

    // ----- Quad (square) transforms -----
    {
        XMMATRIX world = XMMatrixTranslation(0, 0, 1.0f); // behind sphere
        XMMATRIX mvp   = XMMatrixTranspose(world * m_view * m_proj);
        XMStoreFloat4x4(&m_cbMapped[cbOffset + 0].mvp, mvp);
    }

    // ----- Sphere transforms -----
    {
        XMMATRIX world = XMMatrixRotationY(m_rotation);
        XMMATRIX mvp   = XMMatrixTranspose(world * m_view * m_proj);
        XMStoreFloat4x4(&m_cbMapped[cbOffset + 1].mvp, mvp);
    }

    UINT srvSize = m_device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    ThrowIfFailed(m_cmdAlloc[m_frameIndex]->Reset());
    ThrowIfFailed(m_cmdList->Reset(m_cmdAlloc[m_frameIndex].Get(), m_pso.Get()));

    // Viewport & scissor
    D3D12_VIEWPORT vp{ 0, 0, float(WIDTH), float(HEIGHT), 0.0f, 1.0f };
    D3D12_RECT     scissor{ 0, 0, WIDTH, HEIGHT };
    m_cmdList->RSSetViewports(1, &vp);
    m_cmdList->RSSetScissorRects(1, &scissor);

    // Transition RTV
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_renderTargets[m_frameIndex].Get(),
        D3D12_RESOURCE_STATE_PRESENT,
        D3D12_RESOURCE_STATE_RENDER_TARGET);
    m_cmdList->ResourceBarrier(1, &barrier);

    // Set render targets
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
        m_rtvHeap->GetCPUDescriptorHandleForHeapStart(),
        m_frameIndex, m_rtvDescSize);
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle =
        m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    m_cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

    // Clear
    const float clearColor[] = { 0.1f, 0.1f, 0.1f, 1.0f };
    m_cmdList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
    m_cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    // Set pipeline state & root signature
    m_cmdList->SetGraphicsRootSignature(m_rootSig.Get());
    ID3D12DescriptorHeap* heaps[] = { m_srvHeap.Get() };
    m_cmdList->SetDescriptorHeaps(1, heaps);

    m_cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // ----- Draw quad (square) with solid texture -----
    {
        // CBV at offset 0
        D3D12_GPU_VIRTUAL_ADDRESS cbAddr = m_cbResource->GetGPUVirtualAddress()
            + (cbOffset + 0) * sizeof(ConstantBufferData);
        m_cmdList->SetGraphicsRootConstantBufferView(0, cbAddr);

        // SRV slot 0 = solid texture
        CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(
            m_srvHeap->GetGPUDescriptorHandleForHeapStart(), 0, srvSize);
        m_cmdList->SetGraphicsRootDescriptorTable(1, srvHandle);

        m_cmdList->IASetVertexBuffers(0, 1, &m_quadVBV);
        m_cmdList->IASetIndexBuffer(&m_quadIBV);
        m_cmdList->DrawIndexedInstanced(m_quadIndexCount, 1, 0, 0, 0);
    }

    // ----- Draw sphere with checkerboard texture -----
    {
        D3D12_GPU_VIRTUAL_ADDRESS cbAddr = m_cbResource->GetGPUVirtualAddress()
            + (cbOffset + 1) * sizeof(ConstantBufferData);
        m_cmdList->SetGraphicsRootConstantBufferView(0, cbAddr);

        // SRV slot 1 = checkerboard texture
        CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(
            m_srvHeap->GetGPUDescriptorHandleForHeapStart(), 1, srvSize);
        m_cmdList->SetGraphicsRootDescriptorTable(1, srvHandle);

        m_cmdList->IASetVertexBuffers(0, 1, &m_sphereVBV);
        m_cmdList->IASetIndexBuffer(&m_sphereIBV);
        m_cmdList->DrawIndexedInstanced(m_sphereIndexCount, 1, 0, 0, 0);
    }

    // Transition RTV back to present
    barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_renderTargets[m_frameIndex].Get(),
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_PRESENT);
    m_cmdList->ResourceBarrier(1, &barrier);

    ThrowIfFailed(m_cmdList->Close());
}

// ---------------------------------------------------------------------------
void D3D12SphereApp::Render()
{
    RecordCommands();

    ID3D12CommandList* lists[] = { m_cmdList.Get() };
    m_cmdQueue->ExecuteCommandLists(1, lists);

    ThrowIfFailed(m_swapChain->Present(1, 0));
    MoveToNextFrame();
}

// ---------------------------------------------------------------------------
// WinMain
// ---------------------------------------------------------------------------
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int)
{
    // Register window class
    WNDCLASSEX wc{};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = hInst;
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"D3D12SphereApp";
    RegisterClassEx(&wc);

    // Compute window rect for desired client size
    RECT r = { 0, 0, WIDTH, HEIGHT };
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);

    HWND hwnd = CreateWindowEx(0, L"D3D12SphereApp",
        L"D3D12 Sphere - ESC to quit",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        r.right - r.left, r.bottom - r.top,
        nullptr, nullptr, hInst, nullptr);

    ShowWindow(hwnd, SW_SHOW);

    D3D12SphereApp app;
    g_app = &app;
    app.Init(hwnd);

    MSG msg{};
    while (!app.ShouldQuit())
    {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT) goto done;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (app.ShouldQuit()) break;
        app.Render();
    }
done:
    g_app = nullptr;
    return 0;
}
