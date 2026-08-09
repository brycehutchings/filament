# Filament Release Notes log

**If you are merging a PR into main**: please add the release note below, under the *Release notes

**If you are cherry-picking a commit into an rc/ branch**: add the release note under the
appropriate header in [RELEASE_NOTES.md](./RELEASE_NOTES.md).

## Release notes for next branch cut

- vulkan: `VulkanPlatform::SwapChainBundle::depth` is replaced by `depths`, which may hold either a
  single shared depth image or one image per color image.
