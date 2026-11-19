// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Fitted offsets for the PTV1 and P1 models (one table for both), in metres. The SDK's TIP points carry the tip
// extension. A finger's side shift moves its three distal points; side shifts are section midpoints, which agree within
// 0.01 mm over DIP-local Z = 20..25 mm, and the thumb's is the measured PTV1 fingertip sleeve centre. The pads are
// fitted separately.
constexpr std::array<const char*, 2> kFittedModelHashes = {
    "63960181839d5486cc0952bf7a4a5c09b6e9eca78fa4035294a74e13cef32805",
    "ea8e2e3a05091ba9fb229b1d932c612e2740004cfc4262a20471afebf1610d55",
};
constexpr M kPalmBasis = { -4.3368086899420177e-19, -0.99951526706354676,  0.031132473510575541,
                           -0.99592691449259607,    0.0028070342275837352, 0.090120483510145774,
                           -0.090164189062272859,   -0.031005668283909979, -0.99544415591484126 };
constexpr V kPalmPad{ 0.018276321340487119, 0, 0 };
constexpr V kWristPad{ 0.012600000174487119, 0, 0 };
// Per finger in kFingers order; the thumb has no placeholder.
struct FingerFit
{
    V side, proximal_pad, placeholder_pad;
};
constexpr std::array<FingerFit, 5> kFingerFits = { {
    { { 0.0021043985844464278, -0.011934637433200775, -0.0021037908489140032 },
      { 0.016150847149399378, 0.002847830113737538, 0 },
      { 0.016733295878679642, 0.0029505315395890099, 0 } },
    { { 0, -0.012292507172327119, -0.0004292638094403375 }, { 0.016400000000000799, 0, 0 }, { 0.016311316751695868, 0, 0 } },
    { { 0, -0.012253194786528032, 0.0010720156357961448 }, { 0.016399999999997639, 0, 0 }, { 0.017310832018759124, 0, 0 } },
    { { -0.0027006949801747799, -0.011697995150430627, 0.0026745384806665844 },
      { 0.015979669062478725, -0.0036891972912384647, 0 },
      { 0.016417065890189413, -0.0037901783056560599, 0 } },
    { { 0.006935, 0.01201177235048925, 0 }, { 0.010, 0, 0 }, {} },
} };
