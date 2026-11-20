#include "OrderFactory.h"

OrderFactory::OrderFactory(const Config& cfg, std::uint32_t seed)
    : cfg_(cfg),
      rng_(seed),
      urgentRng_(seed ^ 0x85ebca6bu),
      restaurantPick_(0, cfg.restaurants.size() - 1),
      nodePick_(0, cfg.nodes.size() - 1) {}

Order OrderFactory::create(std::int64_t nowMs) {
    const RestaurantConfig& r = cfg_.restaurants[restaurantPick_(rng_)];

    std::string dropoff = cfg_.nodes[nodePick_(rng_)].id;
    while (dropoff == r.node) {
        dropoff = cfg_.nodes[nodePick_(rng_)].id;
    }

    std::uniform_int_distribution<std::int64_t> prep(r.prepMinMs, r.prepMaxMs);

    Order order;
    order.id = nextId_++;
    order.restaurantId = r.id;
    order.pickupNode = r.node;
    order.dropoffNode = dropoff;
    order.state = OrderState::created;
    order.createdMs = nowMs;
    order.prepMs = prep(rng_);
    std::uniform_real_distribution<double> urgentDraw(0.0, 1.0);
    order.urgent = urgentDraw(urgentRng_) < cfg_.urgentProbability;
    return order;
}
