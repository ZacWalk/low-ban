@{
    schema = 1
    project = @{
        name = 'low-ban'
        type = 'gui'
        'default-target' = 'app'
    }
    dependencies = @{ owner = 'dd' }
    build = @{
        'x64-windows' = @{
            debug = 'debug'
            release = 'release'
        }
    }
    targets = @(
        @{
            id = 'app'
            kind = 'gui'
            'cmake-target' = 'low-ban'
            'test-label' = 'low-ban'
            'debug-path' = 'exe/low-ban-64d{exe}'
            'release-path' = 'exe/low-ban-64{exe}'
            platforms = @('x64-windows')
        }
    )
    commands = @{
        evaluate = @{
            description = 'Score the PNG samples in samples/ through the codec offline. No camera needed.'
            script = 'cmake/project-command.ps1'
            effects = 'read'
            'supports-dry-run' = $true
            'timeout-secs' = 1800
            parameters = @{
                config = @{ type = 'string'; default = 'release'; choices = @('debug', 'release') }
            }
        }
    }
}
