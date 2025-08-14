
#ifndef CATCOC_DETAIL_REMOTE_COPY_TYPE_HPP
#define CATCOC_DETAIL_REMOTE_COPY_TYPE_HPP

namespace Catcoc::detail {

enum class CopyMode {P2P, Scatter, Gather};
enum class CopyDirect {Put, Get};

} // namespace Catcoc::detail

#endif // CATCOC_DETAIL_REMOTE_COPY_TYPE_HPP