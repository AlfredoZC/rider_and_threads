#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

#include "Config.h"
#include "CourierActor.h"
#include "Quote.h"
#include "QuotePool.h"
#include "SimContext.h"

class Dispatcher {
public:
    using QuoteFn = std::function<std::int64_t(Router&, const CityGraph&, const Courier&,
                                               const Order&)>;

    Dispatcher(const Config& cfg, const SimContext& ctx,
               const std::vector<std::unique_ptr<CourierActor>>& fleet);

    Dispatcher(const Dispatcher&) = delete;
    Dispatcher& operator=(const Dispatcher&) = delete;

    void run();
    void requestStop();

    void setQuoteFunction(QuoteFn fn) { quote_ = std::move(fn); }

    std::int64_t quotesRequested() const;
    std::size_t quoteWorkers() const { return pool_.workers(); }
    std::int64_t quotesLate() const;

private:
    struct Candidate {
        std::size_t index;
        std::future<std::int64_t> estimate;
    };

    void dispatchOne(OrderId orderId);
    void sweepLateQuotes();
    bool stopped() const;

    bool destinoInalcanzable(const Order& order);

    const SimContext ctx_;
    const std::vector<std::unique_ptr<CourierActor>>& fleet_;
    const std::int64_t quoteTimeoutMs_;
    const std::int64_t acceptTimeoutMs_;
    QuoteFn quote_;

    Router router_;

    QuotePool pool_;

    std::vector<std::future<std::int64_t>> inFlight_;

    mutable std::mutex mutex_;
    bool stopRequested_ = false;
    std::int64_t requested_ = 0;
    std::int64_t late_ = 0;
};
