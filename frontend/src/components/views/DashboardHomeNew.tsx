/**
 * DashboardHome - Empty Dashboard
 * ================================
 * Empty dashboard with just background.
 */

interface DashboardHomeProps {
    isDarkMode: boolean;
}

const DashboardHome: React.FC<DashboardHomeProps> = ({ isDarkMode }) => {
    return (
        <div className="min-h-screen">
            {/* Empty - just background */}
        </div>
    );
};

export default DashboardHome;
