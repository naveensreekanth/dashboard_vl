import { AuthGate } from "@/components/auth/AuthGate";
import { DashboardShell } from "@/components/dashboard/DashboardShell";

export default function HomePage() {
  return (
    <AuthGate>
      <DashboardShell />
    </AuthGate>
  );
}
