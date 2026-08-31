import type { DTLRecommendation } from "@/api/types";

export function getRecommendationForParameter(
  recommendations: DTLRecommendation[],
  parameter: string,
): DTLRecommendation | undefined {
  return recommendations.find((r) => r.parameter === parameter);
}
