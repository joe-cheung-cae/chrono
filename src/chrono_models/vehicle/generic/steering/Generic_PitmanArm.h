// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Radu Serban
// =============================================================================
//
// Generic vehicle Pitman arm steering model.
//
// =============================================================================

#ifndef GENERIC_PITMAN_ARM_H
#define GENERIC_PITMAN_ARM_H

#include "chrono_vehicle/wheeled_vehicle/steering/ChPitmanArm.h"

#include "chrono_models/ChApiModels.h"

namespace chrono {
namespace vehicle {
namespace generic {

/// @addtogroup vehicle_models_generic
/// @{

/// Pitman-arm steering subsystem for the genmeric vehicle.
class CH_MODELS_API Generic_PitmanArm : public ChPitmanArm {
  public:
    Generic_PitmanArm(const std::string& name);
    ~Generic_PitmanArm() {}

    virtual double getSteeringLinkMass() const override { return m_steeringLinkMass; }
    virtual double getPitmanArmMass() const override { return m_pitmanArmMass; }

    virtual double getSteeringLinkRadius() const override { return m_steeringLinkRadius; }
    virtual double getPitmanArmRadius() const override { return m_pitmanArmRadius; }

    virtual const ChVector3d& getSteeringLinkInertiaMoments() const override { return m_steeringLinkInertiaMoments; }
    virtual const ChVector3d& getSteeringLinkInertiaProducts() const override { return m_steeringLinkInertiaProducts; }
    virtual const ChVector3d& getPitmanArmInertiaMoments() const override { return m_pitmanArmInertiaMoments; }
    virtual const ChVector3d& getPitmanArmInertiaProducts() const override { return m_pitmanArmInertiaProducts; }

    virtual double getMaxAngle() const override { return m_maxAngle; }

    virtual const ChVector3d getLocation(PointId which) override;
    virtual const ChVector3d getDirection(DirectionId which) override;

  private:
    static const double m_steeringLinkMass;
    static const double m_pitmanArmMass;

    static const double m_steeringLinkRadius;
    static const double m_pitmanArmRadius;

    static const double m_maxAngle;

    static const ChVector3d m_steeringLinkInertiaMoments;
    static const ChVector3d m_steeringLinkInertiaProducts;
    static const ChVector3d m_pitmanArmInertiaMoments;
    static const ChVector3d m_pitmanArmInertiaProducts;
};

/// @} vehicle_models_generic

}  // end namespace generic
}  // end namespace vehicle
}  // end namespace chrono

#endif
