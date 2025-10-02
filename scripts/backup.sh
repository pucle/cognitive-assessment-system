#!/bin/bash
# Database Backup Script for Cognitive Assessment System
# Creates automated backups of PostgreSQL database

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
BACKUP_DIR="$PROJECT_ROOT/backups"
LOG_FILE="$PROJECT_ROOT/logs/backup.log"

# Default settings
RETENTION_DAYS=30
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILENAME="cognitive_backup_$TIMESTAMP.sql"

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

# Create backup directory
create_backup_dir() {
    if [[ ! -d "$BACKUP_DIR" ]]; then
        mkdir -p "$BACKUP_DIR"
        log_info "Created backup directory: $BACKUP_DIR"
    fi

    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
}

# Validate database connection
validate_db_connection() {
    log_info "Validating database connection..."

    if [[ -z "$DATABASE_URL" ]]; then
        log_error "DATABASE_URL environment variable not set"
        exit 1
    fi

    # Test connection using pg_isready if available
    if command -v pg_isready &> /dev/null; then
        if ! pg_isready "$DATABASE_URL" &> /dev/null; then
            log_error "Cannot connect to database"
            exit 1
        fi
    else
        # Fallback: try to run a simple query
        if ! psql "$DATABASE_URL" -c "SELECT 1;" &> /dev/null; then
            log_error "Cannot connect to database"
            exit 1
        fi
    fi

    log_success "Database connection validated"
}

# Create database backup
create_backup() {
    log_info "Creating database backup..."

    local backup_path="$BACKUP_DIR/$BACKUP_FILENAME"

    # Use pg_dump to create backup
    if ! pg_dump "$DATABASE_URL" --no-owner --no-privileges --clean --if-exists > "$backup_path" 2>> "$LOG_FILE"; then
        log_error "Failed to create database backup"
        exit 1
    fi

    # Compress the backup
    gzip "$backup_path"
    local compressed_backup="$backup_path.gz"

    log_success "Database backup created: $compressed_backup"

    # Get backup size
    local size=$(du -h "$compressed_backup" | cut -f1)
    log_info "Backup size: $size"

    echo "$compressed_backup"
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"
    log_info "Verifying backup integrity..."

    # Decompress and check if it's a valid SQL file
    if ! gunzip -c "$backup_file" | head -n 10 | grep -q "PostgreSQL database dump"; then
        log_error "Backup file appears to be corrupted"
        return 1
    fi

    log_success "Backup integrity verified"
    return 0
}

# Upload backup to cloud storage (optional)
upload_to_cloud() {
    local backup_file="$1"

    # Check if cloud storage is configured
    if [[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" && -n "$AWS_S3_BUCKET" ]]; then
        log_info "Uploading backup to AWS S3..."

        if command -v aws &> /dev/null; then
            if aws s3 cp "$backup_file" "s3://$AWS_S3_BUCKET/backups/" --quiet; then
                log_success "Backup uploaded to S3: s3://$AWS_S3_BUCKET/backups/$(basename "$backup_file")"
                return 0
            else
                log_error "Failed to upload backup to S3"
                return 1
            fi
        else
            log_warning "AWS CLI not installed. Skipping cloud upload."
            return 1
        fi
    elif [[ -n "$CLOUDINARY_URL" ]]; then
        log_info "Cloudinary upload not implemented yet"
        return 1
    else
        log_info "No cloud storage configured. Backup stored locally only."
        return 1
    fi
}

# Clean up old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups (retention: $RETENTION_DAYS days)..."

    local deleted_count=0

    # Find and delete old backups
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]] && [[ $(find "$file" -mtime +$RETENTION_DAYS 2>/dev/null) ]]; then
            rm -f "$file"
            ((deleted_count++))
            log_info "Deleted old backup: $(basename "$file")"
        fi
    done < <(find "$BACKUP_DIR" -name "*.sql.gz" -print0 2>/dev/null)

    if [[ $deleted_count -gt 0 ]]; then
        log_success "Cleaned up $deleted_count old backup(s)"
    else
        log_info "No old backups to clean up"
    fi
}

# Generate backup report
generate_report() {
    local backup_file="$1"
    local success="$2"

    local report_file="$BACKUP_DIR/backup_report_$TIMESTAMP.txt"

    cat << EOF > "$report_file"
Cognitive Assessment System - Backup Report
==========================================

Backup Date: $(date)
Backup File: $(basename "$backup_file")
Backup Size: $(du -h "$backup_file" | cut -f1)
Database URL: ${DATABASE_URL%%@*}@[HIDDEN]
Success: $success

Backup Contents Summary:
$(gunzip -c "$backup_file" | grep -E "^(CREATE TABLE|INSERT INTO|--)" | head -20)

Recent Log Entries:
$(tail -20 "$LOG_FILE" 2>/dev/null || echo "No log file available")

System Information:
- Hostname: $(hostname)
- User: $(whoami)
- Working Directory: $(pwd)
- Free Disk Space: $(df -h . | tail -1 | awk '{print $4}')

==========================================
EOF

    log_info "Backup report generated: $report_file"
}

# Send notification (optional)
send_notification() {
    local success="$1"
    local backup_file="$2"

    # Email notification (if configured)
    if [[ -n "$SMTP_HOST" && -n "$EMAIL_USER" ]]; then
        log_info "Sending email notification..."

        local subject="Cognitive Assessment Backup $(if [[ "$success" == "true" ]]; then echo "Success"; else echo "Failed"; fi)"
        local body="Backup completed at $(date). File: $(basename "$backup_file"). Check logs for details."

        # This would require a mail command or Python script to send email
        log_info "Email notification configured but not implemented yet"
    fi

    # Slack/Discord webhook (if configured)
    if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        local payload="{
            \"text\": \"Cognitive Assessment Backup $(if [[ "$success" == "true" ]]; then echo "✅ Success"; else echo "❌ Failed"; fi)\",
            \"attachments\": [{
                \"fields\": [
                    {\"title\": \"Backup File\", \"value\": \"$(basename "$backup_file")\", \"short\": true},
                    {\"title\": \"Timestamp\", \"value\": \"$TIMESTAMP\", \"short\": true}
                ]
            }]
        }"

        curl -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK_URL" &> /dev/null
        log_info "Slack notification sent"
    fi
}

# Main backup function
main() {
    echo "💾 Cognitive Assessment System - Database Backup"
    echo "==============================================="

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --retention-days)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            --upload)
                UPLOAD_TO_CLOUD=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --retention-days DAYS    Keep backups for DAYS days (default: 30)"
                echo "  --upload                 Upload backup to cloud storage"
                echo "  --dry-run               Show what would be done without doing it"
                echo "  --help, -h              Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Initialize
    create_backup_dir

    log_info "Starting backup process..."
    log_info "Retention days: $RETENTION_DAYS"
    log_info "Upload to cloud: ${UPLOAD_TO_CLOUD:-false}"
    log_info "Dry run: ${DRY_RUN:-false}"

    # Validate database connection
    if [[ "${DRY_RUN:-false}" != "true" ]]; then
        validate_db_connection
    fi

    local backup_file=""
    local success=false

    # Create backup
    if [[ "${DRY_RUN:-false}" != "true" ]]; then
        backup_file=$(create_backup)

        # Verify backup
        if verify_backup "$backup_file"; then
            success=true
            log_success "Backup completed successfully"
        else
            log_error "Backup verification failed"
        fi

        # Upload to cloud if requested
        if [[ "${UPLOAD_TO_CLOUD:-false}" == "true" && "$success" == "true" ]]; then
            if upload_to_cloud "$backup_file"; then
                log_success "Cloud upload completed"
            fi
        fi
    else
        log_info "[DRY RUN] Would create backup: $BACKUP_DIR/$BACKUP_FILENAME.gz"
        success=true
        backup_file="$BACKUP_DIR/$BACKUP_FILENAME.gz"
    fi

    # Clean up old backups
    if [[ "${DRY_RUN:-false}" != "true" ]]; then
        cleanup_old_backups
    fi

    # Generate report
    generate_report "$backup_file" "$success"

    # Send notifications
    if [[ "${DRY_RUN:-false}" != "true" ]]; then
        send_notification "$success" "$backup_file"
    fi

    # Final status
    echo ""
    if [[ "$success" == "true" ]]; then
        echo "✅ BACKUP COMPLETED SUCCESSFULLY!"
        echo "📁 Backup file: $backup_file"
        echo "📊 Check report: $BACKUP_DIR/backup_report_$TIMESTAMP.txt"
        echo "📋 Check logs: $LOG_FILE"
        exit 0
    else
        echo "❌ BACKUP FAILED!"
        echo "📋 Check logs: $LOG_FILE"
        exit 1
    fi
}

# Run main function
main "$@"
