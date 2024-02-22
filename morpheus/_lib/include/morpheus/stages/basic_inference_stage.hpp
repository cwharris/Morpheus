#pragma once

#include "morpheus/export.h"
#include "morpheus/messages/control.hpp"

#include <mrc/segment/builder.hpp>
#include <pymrc/asyncio_runnable.hpp>

namespace morpheus {

class MORPHEUS_EXPORT BasicInferenceStage
  : public mrc::pymrc::AsyncioRunnable<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>
{
  public:
    static std::shared_ptr<mrc::segment::Object<BasicInferenceStage>> init(mrc::segment::Builder& builder,
                                                                            const std::string& name)
    {
        return builder.construct_object<BasicInferenceStage>(name);
    }

  private:
    mrc::coroutines::AsyncGenerator<std::shared_ptr<ControlMessage>> on_data(
        std::shared_ptr<ControlMessage>&& data) override
    {
        co_yield std::move(data);
    }
};

}  // namespace morpheus
