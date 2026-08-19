#include "helloxr_features.h"
#include "helloxr_jetpack_ui.h"
#include "helloxr_quad_layer.h"

#if !defined(__ANDROID__)
#include "generated/resources/resources.h"
#endif

#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>

#include <utils/Entity.h>
#include <utils/EntityManager.h>

#include <math/mat4.h>
#include <math/vec3.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

using namespace filament;
using namespace filament::math;

namespace helloxr {
namespace {

float3 const QUAD_VERTICES[] = {
    { -0.5f, -0.5f, 0.0f },
    { 0.5f, -0.5f, 0.0f },
    { 0.5f, 0.5f, 0.0f },
    { -0.5f, 0.5f, 0.0f },
};

constexpr uint16_t QUAD_INDICES[] = { 0, 1, 2, 0, 2, 3 };

class QuadDepthProxies final : public Feature {
public:
    char const* name() const override { return "quad depth proxies"; }

    std::vector<char const*> requiredExtensions() const override { return {}; }

    bool initialize(FeatureContext const& context) override {
        mContext = context;
        bool const hasJetpack = mContext.jetpackUi != nullptr && mContext.jetpackUi->isEnabled();
        bool const hasFixedQuad = mContext.quadLayer != nullptr && mContext.quadLayer->isEnabled();
        if ((!hasJetpack && !hasFixedQuad) || !loadMaterial()) {
            return false;
        }

        Engine& engine = *mContext.engine;
        mVertices = VertexBuffer::Builder()
                            .vertexCount(4)
                            .bufferCount(1)
                            .attribute(VertexAttribute::POSITION, 0,
                                    VertexBuffer::AttributeType::FLOAT3)
                            .build(engine);
        auto* vertices = new float3[4];
        std::copy(std::begin(QUAD_VERTICES), std::end(QUAD_VERTICES), vertices);
        mVertices->setBufferAt(engine, 0, { vertices, sizeof(QUAD_VERTICES), releaseVertices });

        mIndices = IndexBuffer::Builder()
                           .indexCount(6)
                           .bufferType(IndexBuffer::IndexType::USHORT)
                           .build(engine);
        auto* indices = new uint16_t[6];
        std::copy(std::begin(QUAD_INDICES), std::end(QUAD_INDICES), indices);
        mIndices->setBuffer(engine, { indices, sizeof(QUAD_INDICES), releaseIndices });

        mMaterialInstance = mMaterial->createInstance();
        mMaterialInstance->setColorWrite(false);
        mMaterialInstance->setDepthWrite(true);
        mMaterialInstance->setDepthCulling(true);

        if (hasJetpack && !createProxy(&mJetpackProxy)) {
            return false;
        }
        if (hasFixedQuad && !createProxy(&mFixedQuadProxy)) {
            return false;
        }
        updateTransforms();
        XRLOG("quad depth proxies: Jetpack %s, fixed quad %s", hasJetpack ? "enabled" : "disabled",
                hasFixedQuad ? "enabled" : "disabled");
        return true;
    }

    void update(XrTime) override { updateTransforms(); }

    void terminate() override {
        destroyProxy(&mJetpackProxy);
        destroyProxy(&mFixedQuadProxy);
        if (mMaterialInstance != nullptr) {
            mContext.engine->destroy(mMaterialInstance);
            mMaterialInstance = nullptr;
        }
        if (mVertices != nullptr) {
            mContext.engine->destroy(mVertices);
            mVertices = nullptr;
        }
        if (mIndices != nullptr) {
            mContext.engine->destroy(mIndices);
            mIndices = nullptr;
        }
        if (mMaterial != nullptr) {
            mContext.engine->destroy(mMaterial);
            mMaterial = nullptr;
        }
    }

private:
    static void releaseVertices(void* buffer, size_t, void*) {
        delete[] static_cast<float3*>(buffer);
    }

    static void releaseIndices(void* buffer, size_t, void*) {
        delete[] static_cast<uint16_t*>(buffer);
    }

    bool loadMaterial() {
#if defined(__ANDROID__)
        std::vector<uint8_t> package;
        if (!readAsset("xrDepthProxy.filamat", &package)) {
            XRLOG("quad depth proxies: xrDepthProxy.filamat is missing");
            return false;
        }
        mMaterial =
                Material::Builder().package(package.data(), package.size()).build(*mContext.engine);
#else
        mMaterial = Material::Builder()
                            .package(RESOURCES_XRDEPTHPROXY_DATA, RESOURCES_XRDEPTHPROXY_SIZE)
                            .build(*mContext.engine);
#endif
        return mMaterial != nullptr;
    }

    bool createProxy(utils::Entity* entity) {
        *entity = utils::EntityManager::get().create();
        RenderableManager::Builder(1)
                .boundingBox({ { 0.0f, 0.0f, 0.0f }, { 0.5f, 0.5f, 0.001f } })
                .material(0, mMaterialInstance)
                .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, mVertices, mIndices)
                .layerMask(0xFF, QUAD_DEPTH_PROXY_LAYER)
                .priority(0)
                .culling(false)
                .castShadows(false)
                .receiveShadows(false)
                .build(*mContext.engine, *entity);
        mContext.scene->addEntity(*entity);
        return !entity->isNull();
    }

    void destroyProxy(utils::Entity* entity) {
        if (entity->isNull()) {
            return;
        }
        mContext.scene->remove(*entity);
        mContext.engine->destroy(*entity);
        utils::EntityManager::get().destroy(*entity);
        *entity = {};
    }

    void updateTransforms() {
        auto& transforms = mContext.engine->getTransformManager();
        if (!mJetpackProxy.isNull()) {
            XrPosef const pose = mContext.jetpackUi->getPose();
            transforms.setTransform(transforms.getInstance(mJetpackProxy),
                    mat4f(poseToMat4(pose)) * mat4f::scaling(float3{ JetpackUiLayer::WIDTH_METERS,
                                                  JetpackUiLayer::HEIGHT_METERS, 1.0f }));
        }
        if (!mFixedQuadProxy.isNull()) {
            XrPosef const pose = mContext.quadLayer->getPose();
            XrExtent2Df const size = mContext.quadLayer->getSize();
            transforms.setTransform(transforms.getInstance(mFixedQuadProxy),
                    mat4f(poseToMat4(pose)) *
                            mat4f::scaling(float3{ size.width, size.height, 1.0f }));
        }
    }

    FeatureContext mContext;
    Material* mMaterial = nullptr;
    MaterialInstance* mMaterialInstance = nullptr;
    VertexBuffer* mVertices = nullptr;
    IndexBuffer* mIndices = nullptr;
    utils::Entity mJetpackProxy;
    utils::Entity mFixedQuadProxy;
};

} // anonymous namespace

std::unique_ptr<Feature> createQuadDepthProxies() { return std::make_unique<QuadDepthProxies>(); }

} // namespace helloxr