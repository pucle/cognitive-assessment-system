#!/bin/bash
# Rollback Script for Cognitive Assessment System
# Emergency rollback in case of deployment issues

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/rollback.log"

# Logging
log_info() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Create log directory
create_log_dir() {
    mkdir -p "$(dirname "$LOG_FILE")"
}

# Confirm rollback action
confirm_rollback() {
    local target="$1"
    local action="$2"

    echo "⚠️  ROLLBACK CONFIRMATION REQUIRED"
    echo "=================================="
    echo "Target: $target"
    echo "Action: $action"
    echo ""
    echo "This will:"
    if [[ "$target" == "backend" ]]; then
        echo "  - Rollback backend to previous version on Railway"
        echo "  - Restore previous environment variables"
        echo "  - Restart backend service"
    elif [[ "$target" == "frontend" ]]; then
        echo "  - Rollback frontend to previous deployment on Vercel"
        echo "  - Restore previous environment variables"
    elif [[ "$target" == "database" ]]; then
        echo "  - Restore database from latest backup"
        echo "  - This will overwrite current database state"
    fi
    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " -r
    if [[ ! "$REPLY" =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Rollback cancelled by user"
        exit 0
    fi
}

# Rollback backend on Railway
rollback_backend() {
    log_info "Rolling back backend on Railway..."

    if ! command -v "railway" &> /dev/null; then
        log_error "Railway CLI not found. Please rollback manually in Railway dashboard."
        log_error "Go to: https://railway.app/project/.../deployments"
        exit 1
    fi

    cd "$PROJECT_ROOT"

    # Check if we're in a Railway project
    if [[ ! -f ".railway" ]]; then
        log_error "Not in a Railway project. Run 'railway link' first."
        exit 1
    fi

    # Get current deployment status
    log_info "Checking current deployment status..."
    railway status

    # List recent deployments
    log_info "Recent deployments:"
    railway deployments

    # Rollback to previous deployment
    log_info "Initiating rollback..."
    if railway rollback; then
        log_success "Backend rollback initiated"

        # Wait for rollback to complete
        log_info "Waiting for rollback to complete..."
        sleep 30

        # Check health
        if curl -f "${RAILWAY_URL:-https://your-app.railway.app}/api/health" &> /dev/null; then
            log_success "Backend health check passed after rollback"
        else
            log_warning "Backend health check failed. Please check Railway logs."
        fi

    else
        log_error "Backend rollback failed"
        exit 1
    fi
}

# Rollback frontend on Vercel
rollback_frontend() {
    log_info "Rolling back frontend on Vercel..."

    if ! command -v "vercel" &> /dev/null; then
        log_error "Vercel CLI not found. Please rollback manually in Vercel dashboard."
        log_error "Go to: https://vercel.com/.../deployments"
        exit 1
    fi

    cd "$PROJECT_ROOT/frontend"

    # List recent deployments
    log_info "Recent deployments:"
    vercel ls

    # Rollback to previous deployment
    log_info "Initiating rollback..."
    if vercel rollback; then
        log_success "Frontend rollback initiated"

        # Wait for rollback to complete
        log_info "Waiting for rollback to complete..."
        sleep 30

        # Check if site is accessible
        if curl -f "${VERCEL_URL:-https://your-app.vercel.app}" &> /dev/null; then
            log_success "Frontend is accessible after rollback"
        else
            log_warning "Frontend is not accessible. Please check Vercel logs."
        fi

    else
        log_error "Frontend rollback failed"
        exit 1
    fi
}

# Rollback database from backup
rollback_database() {
    log_info "Rolling back database from backup..."

    # Find latest backup
    local backup_dir="$PROJECT_ROOT/backups"
    local latest_backup=""

    if [[ -d "$backup_dir" ]]; then
        latest_backup=$(find "$backup_dir" -name "*.sql.gz" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    fi

    if [[ -z "$latest_backup" ]]; then
        log_error "No backup files found in $backup_dir"
        log_error "Please ensure backups exist before attempting database rollback"
        exit 1
    fi

    log_info "Found latest backup: $(basename "$latest_backup")"

    # Confirm database rollback (extra confirmation since this is destructive)
    echo ""
    echo "🆘 DATABASE ROLLBACK WARNING"
    echo "============================"
    echo "This will:"
    echo "  - COMPLETELY REPLACE the current database"
    echo "  - All data added since backup will be LOST"
    echo "  - This action CANNOT be undone"
    echo ""
    echo "Backup file: $(basename "$latest_backup")"
    echo "Backup date: $(date -r "$latest_backup")"
    echo ""
    read -p "Type 'YES' to confirm database rollback: " -r
    if [[ "$REPLY" != "YES" ]]; then
        log_info "Database rollback cancelled"
        exit 0
    fi

    # Create pre-rollback backup (just in case)
    log_info "Creating emergency backup before rollback..."
    "$SCRIPT_DIR/backup.sh" --retention-days 1

    # Perform database rollback
    log_info "Restoring database from backup..."

    # Decompress and restore
    if gunzip -c "$latest_backup" | psql "$DATABASE_URL" 2>> "$LOG_FILE"; then
        log_success "Database rollback completed successfully"
    else
        log_error "Database rollback failed"
        log_error "Check the logs and consider restoring from emergency backup"
        exit 1
    fi

    # Verify database integrity
    log_info "Verifying database integrity..."
    if psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM users;" &> /dev/null; then
        log_success "Database integrity check passed"
    else
        log_error "Database integrity check failed"
        exit 1
    fi
}

# Rollback environment variables
rollback_env_vars() {
    local target="$1"
    log_info "Rolling back environment variables for $target..."

    if [[ "$target" == "backend" ]]; then
        if command -v "railway" &> /dev/null; then
            log_info "Please check Railway dashboard for environment variable history"
            log_info "Railway does not support automated env var rollback via CLI"
        fi
    elif [[ "$target" == "frontend" ]]; then
        if command -v "vercel" &> /dev/null; then
            log_info "Please check Vercel dashboard for environment variable history"
            log_info "Vercel does not support automated env var rollback via CLI"
        fi
    fi

    log_warning "Environment variable rollback must be done manually in the dashboard"
}

# Full system rollback
rollback_full_system() {
    log_info "Performing full system rollback..."

    # Rollback in reverse order: frontend first, then backend, then database
    log_info "Step 1: Rolling back frontend..."
    rollback_frontend

    log_info "Step 2: Rolling back backend..."
    rollback_backend

    log_info "Step 3: Rolling back database..."
    rollback_database

    log_success "Full system rollback completed"
}

# Generate rollback report
generate_rollback_report() {
    local target="$1"
    local success="$2"

    local report_file="$PROJECT_ROOT/logs/rollback_report_$(date +%Y%m%d_%H%M%S).txt"

    cat << EOF > "$report_file"
Cognitive Assessment System - Rollback Report
==========================================

Rollback Date: $(date)
Target: $target
Success: $success

System Status After Rollback:
- Backend URL: ${RAILWAY_URL:-'Unknown'}
- Frontend URL: ${VERCEL_URL:-'Unknown'}
- Database: ${DATABASE_URL:+Connected}

Recent Log Entries:
$(tail -20 "$LOG_FILE" 2>/dev/null || echo "No log entries available")

Troubleshooting Steps:
1. Check Railway/Vercel dashboards for deployment status
2. Verify environment variables are correct
3. Test API endpoints: curl https://your-backend.railway.app/api/health
4. Check application logs for errors
5. If issues persist, consider deploying from a known good commit

Emergency Contacts:
- Railway Support: https://railway.app/support
- Vercel Support: https://vercel.com/support
- Check DEPLOYMENT_SUMMARY.md for service URLs

==========================================
EOF

    log_info "Rollback report generated: $report_file"
}

# Show help
show_help() {
    cat << EOF
Cognitive Assessment System - Rollback Script

USAGE:
    $0 [TARGET] [OPTIONS]

TARGETS:
    backend     Rollback backend on Railway
    frontend    Rollback frontend on Vercel
    database    Restore database from backup
    full        Rollback entire system (frontend → backend → database)

OPTIONS:
    --confirm   Skip confirmation prompts
    --help, -h  Show this help message

EXAMPLES:
    $0 backend              # Rollback backend only
    $0 frontend             # Rollback frontend only
    $0 database             # Restore database from backup
    $0 full                 # Full system rollback
    $0 backend --confirm    # Rollback backend without confirmation

WARNING:
    Database rollback is DESTRUCTIVE and will overwrite current data.
    Always ensure you have recent backups before proceeding.

LOGS:
    All rollback actions are logged to: $LOG_FILE
EOF
}

# Main rollback function
main() {
    echo "🔄 Cognitive Assessment System - Rollback Script"
    echo "==============================================="

    # Initialize
    create_log_dir

    # Parse arguments
    local target=""
    local skip_confirm=false

    if [[ $# -eq 0 ]]; then
        show_help
        exit 1
    fi

    case "$1" in
        backend|frontend|database|full)
            target="$1"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Invalid target: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac

    # Parse options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --confirm)
                skip_confirm=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Confirmation
    if [[ "$skip_confirm" != "true" ]]; then
        case "$target" in
            backend)
                confirm_rollback "backend" "Rollback to previous Railway deployment"
                ;;
            frontend)
                confirm_rollback "frontend" "Rollback to previous Vercel deployment"
                ;;
            database)
                # Database confirmation is handled in rollback_database()
                ;;
            full)
                confirm_rollback "full system" "Complete system rollback (frontend, backend, database)"
                ;;
        esac
    fi

    # Perform rollback
    local success=false

    case "$target" in
        backend)
            if rollback_backend; then
                success=true
            fi
            ;;
        frontend)
            if rollback_frontend; then
                success=true
            fi
            ;;
        database)
            if rollback_database; then
                success=true
            fi
            ;;
        full)
            if rollback_full_system; then
                success=true
            fi
            ;;
    esac

    # Generate report
    generate_rollback_report "$target" "$success"

    # Final status
    echo ""
    if [[ "$success" == "true" ]]; then
        echo "✅ ROLLBACK COMPLETED SUCCESSFULLY!"
        echo "📋 Check report: $PROJECT_ROOT/logs/rollback_report_*.txt"
        echo "📊 Monitor your applications in Railway/Vercel dashboards"
    else
        echo "❌ ROLLBACK FAILED!"
        echo "📋 Check logs: $LOG_FILE"
        echo "🔧 Manual intervention may be required"
        exit 1
    fi
}

# Run main function
main "$@"
